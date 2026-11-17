import copy
import unittest
from flightmare_profile.diagnose import attribute, stability


class AttributionTests(unittest.TestCase):
    def test_stage_drift_is_visible_even_when_total_wall_is_stable(self):
        a=attribute(self.fixture())[0]
        a.update(run=0,anchor_repeat=False)
        b=copy.deepcopy(a); b.update(run=1,anchor_repeat=True)
        b['stages']['dynamics']['us']*=1.3
        result=stability([b,a])[6]
        self.assertEqual(result['wall_ratio'],1)
        self.assertFalse(result['stage_attribution_stable'])

    def fixture(self):
        return dict(status='passed',config=dict(iterations=1000,control_dt=.01,reset_percent=50),cases=[dict(
            environments=2,slots=3,timing=dict(wall_us_per_iteration=dict(median=10,min=9,max=11)),
            native=dict(auto_resets_including_warmup=8,final_done_lanes=2),
            trace=dict(cuda_api_errors=0,kernel_span_us=2560,idle_gap_fraction_between_first_last_kernel=.1,
                       kernels=[dict(name=k,count=256,total_us=t,max_registers_per_thread=32)
                                for k,t in [('prepareActions',256),('dynamicsKernel',512),('taskKernel',768)]]))])

    def test_trace_window_normalization_and_stage_share(self):
        row=attribute(self.fixture())[0]
        self.assertEqual(row['lanes'],6)
        self.assertEqual(row['stages']['dynamics']['us'],2)
        self.assertAlmostEqual(row['stages']['task_reset']['share'],.5)
        self.assertAlmostEqual(row['stages']['task_reset']['ideal_total_speedup_if_halved'],4/3)

    def test_missing_failed_or_ambiguous_evidence_is_rejected(self):
        base=self.fixture()
        for mutation in ('failed','count','names','nan','missing'):
            report=copy.deepcopy(base)
            trace=report['cases'][0]['trace']
            if mutation=='failed': report['status']='failed'
            if mutation=='count': trace['kernels'][0]['count']=255
            if mutation=='names': trace['kernels'][0]['name']='taskKernel'
            if mutation=='nan': trace['kernels'][0]['total_us']=float('nan')
            if mutation=='missing': del report['cases'][0]['trace']
            with self.assertRaises(ValueError): attribute(report)
