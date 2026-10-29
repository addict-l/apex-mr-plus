import unittest
import numpy as np
from normal.program import compress_collinear,parameterize
from normal.tpg import build_tpg

class DynamicReferenceTests(unittest.TestCase):
    def test_compression_preserves_corner(self):
        path=[[0]*6,[.1,0,0,0,0,0],[.2,0,0,0,0,0],[.2,.1,0,0,0,0]]
        self.assertEqual(compress_collinear(path),[path[0],path[2],path[3]])

    def test_quintic_velocity_and_acceleration_limits(self):
        graph=build_tpg([{'id':'t','arm':'A','kind':'transit','duration':1.,
                         'q_path':[[0]*6,[.24,-.12,0,0,0,0]]}],max_joint_span=1.)
        parameterize(graph,vmax=.6,amax=1.2)
        node=graph['nodes'][0]
        segment=node['motion_segments'][0];t=np.linspace(0,segment['duration'],2001)
        u=t/segment['duration'];s=10*u**3-15*u**4+6*u**5
        q=s[:,None]*(np.array(segment['q1'])-segment['q0'])
        v=np.gradient(q,t,axis=0);a=np.gradient(v,t,axis=0)
        self.assertLessEqual(np.max(np.abs(v)),.60001)
        self.assertLessEqual(np.max(np.abs(a)),1.20001)
        self.assertLess(np.max(np.abs(v[[0,-1]])),1e-5)

    def test_contact_skill_obeys_slower_velocity_limit(self):
        graph=build_tpg([{'id':'support','arm':'A','kind':'support_start','duration':1.,
                         'q_path':[[0]*6,[.08,-.01,0,0,0,0]],
                         'metadata':{'max_velocity_rad_s':.025,'max_acceleration_rad_s2':.12}}])
        parameterize(graph,vmax=.6,amax=1.2)
        for node in graph['nodes']:
            for segment in node['motion_segments']:
                t=np.linspace(0,segment['duration'],2001);u=t/segment['duration']
                q=(10*u**3-15*u**4+6*u**5)[:,None]*(np.array(segment['q1'])-segment['q0'])
                velocity=np.gradient(q,t,axis=0)
                acceleration=np.gradient(velocity,t,axis=0)
                self.assertLessEqual(np.max(np.abs(velocity)),.025001)
                self.assertLessEqual(np.max(np.abs(acceleration)),.120001)

if __name__=='__main__':unittest.main()
