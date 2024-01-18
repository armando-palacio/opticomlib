import sys; sys.path.append(sys.path[0]+"\..") 

import unittest
import _types_ as tp
import numpy as np

class Test_binary_sequence(unittest.TestCase): 
    def test_binary_sequence_class_(self):
        cases = [tp.binary_sequence('01101101'), 
              tp.binary_sequence([0,1,1,0,1,1,0,1]), 
              tp.binary_sequence((0,1,1,0,1,1,0,1)),
              tp.binary_sequence(np.array((0,1,1,0,1,1,0,1)))]
        
        for bs in cases:
            self.assertEqual(type(bs.data), np.ndarray)
            self.assertEqual(bs.len(),8)
            self.assertEqual(bs.__len__(),8)
            self.assertEqual(bs.type(), tp.binary_sequence)


class Test_electrical_signal(unittest.TestCase):
    def test_electrical_signal_class(self):
        input = np.array([.1,-.2,.2])
        
        cases = [tp.electrical_signal(input.tolist()),
                 tp.electrical_signal(tuple(input)),
                 tp.electrical_signal(input)]
        
        for e in cases:
            self.assertEqual(type(e.signal), np.ndarray)
            self.assertEqual(type(e.noise), np.ndarray)
            self.assertEqual(e.len(),input.size)
            self.assertEqual(e.__len__(),input.size)
            self.assertEqual(e.type(), tp.electrical_signal)
            self.assertEqual(e.print(), e)
            self.assertEqual(e.fs(), tp.global_vars.fs)
            self.assertEqual(e.sps(), tp.global_vars.sps)
            self.assertEqual(e.dt(), tp.global_vars.dt)
            
            self.assertTrue(np.array_equal(e.abs('signal'), np.abs(input)))
            self.assertTrue(np.array_equal(e.abs('noise'), np.zeros_like(input)))
            self.assertTrue(np.array_equal(e.abs('all'), np.abs(input)))
            self.assertTrue(np.array_equal(e.abs(), np.abs(input)))

            self.assertTrue(e.power('signal') == np.mean(input**2))
            self.assertTrue(e.power('noise') == 0)
            self.assertTrue(e.power('all') == np.mean(input**2))
            self.assertTrue(e.power() == np.mean(input**2))

            self.assertNotEqual(e.copy(), e)
            self.assertEqual(e.copy(2).len(), 2)
            self.assertTrue(np.array_equal(e.copy(2).signal, input[:2]))

            self.assertTrue(np.array_equal(e.__add__(4).signal, input+4))
            self.assertTrue(np.array_equal(e.__add__(4).noise, np.ones_like(input)*4))
            self.assertTrue(np.array_equal(e.__add__(e).signal, input+input))
            self.assertTrue(np.array_equal(e.__add__(e).noise, np.zeros_like(input)))

            self.assertTrue(np.array_equal(e.__mul__(4).signal, input*4))
            self.assertTrue(np.array_equal(e.__mul__(4).noise, np.zeros_like(input)))

            self.assertTrue(np.array_equal(e.__mul__(e).signal, input**2))
            self.assertTrue(np.array_equal(e.__mul__(e).noise, np.zeros_like(input)))

            self.assertTrue(np.array_equal(e.__getitem__(slice(0,2)).signal, input[:2]))
            self.assertTrue(np.array_equal(e.__getitem__(slice(0,2)).noise, np.zeros(2)))

            self.assertTrue(np.array_equal(e.__call__('w').signal, tp.fft(input)))
            self.assertTrue(np.array_equal(e.__call__('t').signal, tp.ifft(input)))


class Test_optical_signal(unittest.TestCase):
    def test_electrical_signal_class(self):
        input = np.array([[.1,-.2,.2],[.1,-.2,.2]])

        cases = [tp.optical_signal(input.tolist()),
                 tp.optical_signal(tuple(input.tolist())),
                 tp.optical_signal(input)]
        
        for e in cases:
            self.assertEqual(type(e.signal), np.ndarray)
            self.assertEqual(type(e.noise), np.ndarray)
            self.assertEqual(e.len(), input.shape[1])
            self.assertEqual(e.type(), tp.optical_signal)
            
            self.assertTrue(np.array_equal(e.abs('signal'), np.abs(input)))
            self.assertTrue(np.array_equal(e.abs('noise'), np.zeros_like(input)))
            self.assertTrue(np.array_equal(e.abs('all'), np.abs(input)))
            self.assertTrue(np.array_equal(e.abs(), np.abs(input)))

            
            self.assertTrue(np.all(e.power('signal') == np.mean(input**2,axis=-1)))
            self.assertTrue(np.all(e.power('noise') == 0))
            self.assertTrue(np.all(e.power('all') == np.mean(input**2,axis=-1)))
            self.assertTrue(np.all(e.power() == np.mean(input**2,axis=-1)))

            self.assertTrue(np.array_equal(e.__add__(4).signal, input+4))
            self.assertTrue(np.array_equal(e.__add__(4).noise, 4*np.ones_like(input)))
            self.assertTrue(np.array_equal(e.__add__(e).signal, 2*input))
            self.assertTrue(np.array_equal(e.__add__(e).noise, np.zeros_like(input)))

            self.assertTrue(np.array_equal(e.__mul__(4).signal, input*4))
            self.assertTrue(np.array_equal(e.__mul__(4).noise, np.zeros_like(input)))
            self.assertTrue(np.array_equal(e.__mul__(e).signal, input**2))
            self.assertTrue(np.array_equal(e.__mul__(e).noise, np.zeros_like(input)))

            self.assertTrue(np.array_equal(e.__getitem__(slice(0,2)).signal, input[:,0:2]))
            self.assertTrue(np.array_equal(e.__getitem__(slice(0,2)).noise, np.zeros_like(input)[:,0:2]))

            self.assertTrue(np.array_equal(e.__call__('w').signal, tp.fft(input)))
            self.assertTrue(np.array_equal(e.__call__('t').signal, tp.ifft(input)))     


class Test_str_to_list_funcion(unittest.TestCase):
    def test_str_to_list(self):
        cases = ['1234', '1,2,3,4', '1 2 3 4', '12 34']

        for e in cases:        
            self.assertTrue(np.array_equal(tp.str_to_list(e), [1,2,3,4]))
            

class Test_db_funcion(unittest.TestCase):
    def test_db(self):
        cases = [10,100,1000]
        for e in cases:
            self.assertTrue(tp.db(e)==10*np.log10(e))

class Test_idb_funcion(unittest.TestCase):
    def test_db(self):
        cases = [0,3,6]
        for e in cases:
            self.assertTrue(tp.idb(e)==10**(e/10.0))

class Test_dbm_funcion(unittest.TestCase):
    def test_dbm(self):
        cases = [1e-3,1e-2,1e-1]
        for e in cases:
            self.assertTrue(tp.dbm(e)==10*np.log10(e/1e-3))

class Test_idbm_funcion(unittest.TestCase):
    def test_dbm(self):
        cases = [0,10,20]
        for e in cases:
            self.assertTrue(tp.idbm(e)==1e-3*10**(e/10.0))

            
if __name__=='__main__':
    unittest.main()