from sanic import Sanic
from sanic.response import json
from sanic.exceptions import NotFound
from peewee_async import Manager
from peewee import operator, reduce
import asyncio
import uvloop
import logging

from universe.settings import LOAD_MODEL_ENDPOINTS
from models.entries import postgres_database
from universe.main_endpoints import search_route

logging.basicConfig(format='%(asctime)s:%(message)s',filename='universe.log',level=logging.INFO)
uvloop.install()

app = Sanic(__name__)
# CORS(app)
loop = asyncio.get_event_loop()

@app.middleware('request')
async def handle_request(request):
	try:
		postgres_database.connection()
	except:
		pass

@app.middleware('response')
async def handle_response(request, response):
	# request.app.queue.put_nowait(request) # update user credential token remaining
	if not postgres_database.is_closed():
		postgres_database.close()

app.blueprint(search_route)

if LOAD_MODEL_ENDPOINTS and LOAD_MODEL_ENDPOINTS is not None:
    from .model_endpoints import model_routes
    app.blueprint(model_routes)    


