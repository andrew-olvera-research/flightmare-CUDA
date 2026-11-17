import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from flightmare_profile.__main__ import summarize, trace_summary, performance_gate, launches, validate_memory_trace
from flightmare_profile.cpu_parallel import validate


class EvidenceTests(unittest.TestCase):
    def result(self):
        return {'validated': True, 'samples': [{'submit_ms': 1, 'wall_ms': 2, 'gpu_timeline_ms': 1.8}],
                'calls': {'allocations': 0, 'frees': 0, 'copies': 0, 'host_syncs': 0, 'kernel_launches': 10}}

    def test_units_and_call_normalization(self):
        out = summarize(self.result(), 10, 1, 'state')
        self.assertEqual(out['timing']['wall_us_per_iteration']['median'], 200)
        self.assertEqual(out['calls_per_iteration']['kernel_launches'], 1)

    def test_missing_work_or_hot_path_copy_cannot_pass(self):
        for field in ('copies', 'host_syncs', 'allocations', 'frees', 'kernel_launches'):
            data = self.result()
            data['calls'][field] += 1
            with self.assertRaises(ValueError):
                summarize(data, 10, 1, 'state')
        data = self.result(); data['validated'] = False
        with self.assertRaises(ValueError):
            summarize(data, 10, 1, 'state')

    def test_trace_gaps_are_not_kernel_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.sqlite'
            with sqlite3.connect(path) as db:
                db.executescript('''
                create table StringIds(id integer,value text);
                insert into StringIds values(1,'kernel'),(2,'cudaLaunchKernel');
                create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,nameId integer,returnValue integer);
                insert into CUPTI_ACTIVITY_KIND_RUNTIME values(0,100,2,0);
                create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,demangledName integer,registersPerThread integer,localMemoryPerThread integer);
                insert into CUPTI_ACTIVITY_KIND_KERNEL values(1000,2000,1,32,0),(5000,6000,1,32,0);
                ''')
            out = trace_summary(path)
            self.assertEqual(out['kernel_active_us'], 2)
            self.assertEqual(out['kernel_span_us'], 5)
            self.assertAlmostEqual(out['idle_gap_fraction_between_first_last_kernel'], .6)
            self.assertEqual(out['memory_events']['CUPTI_ACTIVITY_KIND_MEMCPY'], 0)

            # Concurrent streams overlap: summed time must not produce a negative
            # idle fraction or be presented as occupied timeline duration.
            with sqlite3.connect(path) as db:
                db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(1500,5500,1,32,0)')
            overlapping = trace_summary(path)
            self.assertEqual(overlapping['kernel_active_us'], 6)
            self.assertEqual(overlapping['kernel_union_us'], 5)
            self.assertEqual(overlapping['idle_gap_fraction_between_first_last_kernel'], 0)

    def test_missing_trace_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                trace_summary(Path(directory) / 'empty.sqlite')


class PerformanceGateTests(unittest.TestCase):
    def report(self):
        return {'status': 'passed', 'gpu_before': 'header\nGPU, UUID, driver, 16000, 30, 1000, 25\n',
                'cuda_architectures': '120-real', 'cuda': '13.3', 'compiler': 'gcc', 'config': {},
                'cases': [{'native': {}, 'environments': 3, 'slots': 11, 'workload': 'environment',
                           'timing': {'wall_us_per_iteration': {'median': 20}}}]}

    def test_regression_and_mismatched_evidence_fail(self):
        baseline = self.report()
        for change in ('slow', 'config', 'hardware', 'missing', 'failed', 'policy'):
            current = self.report(); old = copy.deepcopy(baseline)
            if change == 'slow': current['cases'][0]['timing']['wall_us_per_iteration']['median'] = 24
            if change == 'config': current['config']['reset_percent'] = 50
            if change == 'hardware': current['cuda_architectures'] = 'other'
            if change == 'missing': old['cases'] = []
            if change == 'failed': old['status'] = 'failed'
            if change == 'policy': current['cases'][0]['native']['policy_workload_version'] = 2
            with self.assertRaises(ValueError): performance_gate(current, old, 1.15)
        current = self.report(); performance_gate(current, baseline, 1.15)
        self.assertTrue(current['performance_gate']['passed'])

    def test_environment_launch_budgets(self):
        self.assertEqual(launches('environment'), 3)
        self.assertEqual(launches('environment-split'), 4)
        self.assertEqual(launches('environment-random'), 3)
        data = EvidenceTests().result(); data['calls']['kernel_launches'] = 30
        summarize(data, 10, 1, 'environment')
        with self.assertRaises(ValueError): summarize(data, 10, 1, 'environment-split')


class GraphEvidenceTests(unittest.TestCase):
    def test_parallel_comparison_rejects_behavior_mismatch(self):
        reference={'output':[0.0]*26,'resets':10}
        validate(reference,copy.deepcopy(reference))
        for index,value in ((2,.1),(24,float('nan')),(25,1)):
            actual=copy.deepcopy(reference);actual['output'][index]=value
            with self.assertRaises(ValueError): validate(reference,actual)
        actual=copy.deepcopy(reference);actual['resets']=9
        with self.assertRaises(ValueError): validate(reference,actual)
        actual=copy.deepcopy(reference);worker=copy.deepcopy(reference);worker['output'][0]=1
        actual['worker_outputs']=[worker]
        with self.assertRaises(ValueError): validate(reference,actual)

    def test_policy_scratch_budget_never_allows_transfers(self):
        trace = {'memory_events': {'CUPTI_ACTIVITY_KIND_MEMCPY': 0, 'CUPTI_ACTIVITY_KIND_MEMSET': 10},
                 'memset_bytes': 10240, 'memset_nonzero_events': 0}
        validate_memory_trace(trace, 'policy-graph', 10)
        with self.assertRaises(ValueError): validate_memory_trace(trace, 'python-graph', 10)
        for field, value in (('memset_bytes', 10241), ('memset_nonzero_events', 1)):
            bad = copy.deepcopy(trace); bad[field] = value
            with self.assertRaises(ValueError): validate_memory_trace(bad, 'policy-graph', 10)
        trace['memory_events']['CUPTI_ACTIVITY_KIND_MEMCPY'] = 1
        with self.assertRaises(ValueError): validate_memory_trace(trace, 'policy-graph', 10)

    def test_graph_api_and_kernel_nodes_are_distinct(self):
        data = EvidenceTests().result()
        data['calls']['kernel_launches'] = 0
        data['calls']['graph_launches'] = 10
        summarize(data, 10, 1, 'environment-graph')
        data['calls']['graph_launches'] = 9
        with self.assertRaises(ValueError): summarize(data, 10, 1, 'environment-graph')

    def test_python_allocator_changes_fail(self):
        data = EvidenceTests().result()
        del data['calls']
        data['allocator_delta'] = {k: 0 for k in ('num_device_alloc', 'num_device_free', 'allocation.all.allocated', 'allocation.all.freed', 'allocated_bytes.all.current', 'reserved_bytes.all.current')}
        summarize(data, 10, 1, 'policy-graph')
        data['allocator_delta']['allocation.all.allocated'] = 1
        with self.assertRaises(ValueError): summarize(data, 10, 1, 'policy-graph')
