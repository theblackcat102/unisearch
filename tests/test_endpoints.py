import os, sys
if os.path.join(os.getcwd(), 'backend') in sys.path:
    sys.path.remove(os.path.join(os.getcwd(), 'backend'))

sys.path.insert(0, os.path.join(os.getcwd(), 'frontend'))

import asyncio
import unittest
import importlib
import universe

importlib.reload(universe)
try: # setup for executing all at once tests, reset settings value from other ends
    importlib.reload(universe.settings)
except AttributeError:
    pass

from universe.app import app





class FrondendAPITest(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(None)

    def test_privacy(self):
        request, response = app.test_client.get('/privacy')
        self.assertEqual(response.status, 200)