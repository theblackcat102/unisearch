import os, sys
sys.path.append(os.path.join(os.getcwd(), 'backend'))


import unittest
from time import time
import numpy as np
import torch
from mclip.utils import np_cache

class FakeTokenizer():
    def __call__(self, context, **kwargs):
        return [0,1,1,1,1,1]

class FakeMCLIP():

    def encode_text2text_text(self, text_tensor):
        return torch.from_numpy(np.array([[1,2,3,4,5,6]]))

    def encode_text2image_text(self, text_tensor):
        return torch.from_numpy(np.array([[1,2,3,4,5,6]]))

    def encode_image(self, img_tensor):
        return torch.from_numpy(np.array([[1,2,3,4,5,6]]))

class FakeTransform():
    def __call__(self, image):
        return torch.from_numpy(np.array([[1,2,3,4,5,6]]))

from mclip import functs

functs.tokenizer = FakeTokenizer()
functs.model = FakeMCLIP()
functs.img_transforms = FakeTransform()


class VectorCache(unittest.TestCase):
    '''
        Ensure cache is working
    '''
    def test_cache(self):

        @np_cache(maxsize=100)
        def test_f(x):
            return x + 1

        start = time()

        self.assertTrue((test_f(np.zeros(10)) == np.ones(10)).all() )
        self.assertTrue((test_f(np.ones(10)) == np.ones(10)*2).all() )
        self.assertTrue((test_f(np.ones(10)*2) == np.ones(10)*3).all() )
        t1 = time() - start


        start = time()
        self.assertTrue((test_f(np.zeros(10)) == np.ones(10)).all() )
        self.assertTrue((test_f(np.ones(10)) == np.ones(10)*2).all() )
        self.assertTrue((test_f(np.ones(10)*2) == np.ones(10)*3).all() )
        t2 = time() - start
        self.assertTrue(t1 > t2)


    def test_text2text_retrieve_encodings(self):
        start = time()
        self.assertTrue(isinstance(functs.text2text_retrieve_encodings('Hello World'), list))
        t1 = time() - start


        start = time()
        for _ in range(10):
            self.assertTrue(isinstance(functs.text2text_retrieve_encodings('Hello World'), list))
        t2 = ( time() - start)/10
        self.assertTrue(t1 > t2)

    def test_img2text_retrieve_encoding_by_img(self):

        start = time()
        self.assertTrue(isinstance(functs.img2text_retrieve_encoding_by_img(np.zeros((1, 3, 32,32))), list))
        t1 = time() - start


        start = time()
        for _ in range(10):
            self.assertTrue(isinstance(functs.img2text_retrieve_encoding_by_img(np.zeros((1, 3, 32, 32))), list))
        t2 = (time() - start)/10
        self.assertTrue(t1 > t2)


    def test_img2text_retrieve_encoding_by_text(self):
        start = time()
        self.assertTrue(isinstance(functs.img2text_retrieve_encoding_by_text('Hello World'), list))
        t1 = time() - start


        start = time()
        self.assertTrue(isinstance(functs.img2text_retrieve_encoding_by_text('Hello World'), list))
        t2 = time() - start
        self.assertTrue(t1 > t2)
