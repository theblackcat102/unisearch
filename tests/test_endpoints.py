import os, sys
sys.path.append(os.path.join(os.getcwd(), 'frontend'))

import asyncio
import unittest
from universe.app import app





class FrondendAPITest(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(None)

    def test_privacy(self):
        request, response = app.test_client.get('/privacy')
        self.assertEqual(response.status, 200)