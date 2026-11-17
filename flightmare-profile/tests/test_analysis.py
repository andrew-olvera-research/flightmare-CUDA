import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from flightmare_profile.__main__ import summarize, trace_summary, performance_gate, launches


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

    def test_missing_trace_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                trace_summary(Path(directory) / 'empty.sqlite')


class PerformanceGateTests(unittest.TestCase):
    def report(self):
        return {'status': 'passed', 'gpu_before': 'header\nGPU, UUID, driver, 16000, 30, 1000, 25\n',
                'cuda_architectures': '120-real', 'cuda': '13.3', 'compiler': 'gcc', 'config': {},
                'cases': [{'environments': 3, 'slots': 11, 'workload': 'environment',
                           'timing': {'wall_us_per_iteration': {'median': 20}}}]}

    def test_regression_and_mismatched_evidence_fail(self):
        baseline = self.report()
        for change in ('slow', 'config', 'hardware', 'missing', 'failed'):
            current = self.report(); old = copy.deepcopy(baseline)
            if change == 'slow': current['cases'][0]['timing']['wall_us_per_iteration']['median'] = 24
            if change == 'config': current['config']['reset_percent'] = 50
            if change == 'hardware': current['cuda_architectures'] = 'other'
            if change == 'missing': old['cases'] = []
            if change == 'failed': old['status'] = 'failed'
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
