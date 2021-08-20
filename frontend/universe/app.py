from sanic import Sanic
from sanic import response
from sanic.response import redirect
from sanic_jinja2 import SanicJinja2
from sanic_session import Session, InMemorySessionInterface
import logging
import aiohttp
import html
from urllib.parse import unquote
import json
from universe.settings import (
    IMG_ML_DOMAIN,
    ML_DOMAIN, SEARCH_TEXT, TARGET_VECTOR_SIZE,
    SEARCH_IMG_BY_TEXT, 
    SEARCH_ENCODED_IMG,
    IMAGE_CDN, ENCODED_IMG
)

logging.basicConfig(format='%(asctime)s:%(message)s',filename='frontend.log',level=logging.INFO)

app = Sanic(__name__)
app.static('/static', './universe/static')

session = Session(app, interface=InMemorySessionInterface())
jinja = jinja = SanicJinja2(app, pkg_name='universe', pkg_path='templates', 
    session=session)


@app.listener('before_server_start')
def init(app, loop):
    app.aiohttp_session = aiohttp.ClientSession(loop=loop)

@app.listener('after_server_stop')
def finish(app, loop):
    loop.run_until_complete(app.aiohttp_session.close())
    loop.close()


@app.route("/search", methods=["POST","GET"])
async def search_text(request):
    if 'q' not in request.args:
        return redirect('/')
    if request.method == 'POST':
        if 'query' in request.form:
            query_text = request.form.get('query')
            url = app.url_for('search_text', q=query_text)
            return redirect(url)

    query_text = request.args['q'][0]
    if isinstance(query_text, str):
        url = ML_DOMAIN+SEARCH_TEXT
        async with app.aiohttp_session.post(url, data={'context': query_text}) as response:
            result = await response.json()
            if result['success']:
                results = result['results']
                for idx in range(len(results)):
                    results[idx]['url_text'] = unquote(results[idx]['url'])
                    results[idx]['description'] = html.escape(results[idx]['description'])
                img_url = request.url.replace('/search', '/search-image')
                return jinja.render("text_results.html", request, results=results, text=query_text, img_url=img_url)
    return redirect('/')

@app.route("/search-image", methods=["POST","GET"])
async def search_image(request):
    if 'q' not in request.args and 'v' not in request.args:
        return redirect('/')
    text_url = request.url.replace('/search-image', '/search') 
    if request.method == 'POST':
        if 'image_query' in request.form:
            query_text = request.form.get('image_query')
            url = app.url_for('search_image', q=query_text)
            return redirect(url)
    if 'v' in request.args:
        vector_raw = request.args['v'][0]
        vector = [float(v) for v in vector_raw.split(',')]
        if len(vector) != TARGET_VECTOR_SIZE:
            return redirect('/')

        url = ML_DOMAIN+SEARCH_ENCODED_IMG

        async with app.aiohttp_session.post(url, data=json.dumps({'vector': vector})) as response:
            result = await response.json()
            if result['success']:
                results = result['results']

                for idx in range(len(results)):
                    results[idx]['cdn_img'] = IMAGE_CDN+results[idx]['image_hash']
                return jinja.render("img_results.html", request, results=results, text='vector', text_url=text_url)


    query_text = request.args['q'][0]
    if isinstance(query_text, str):
        url = ML_DOMAIN+SEARCH_IMG_BY_TEXT
        async with app.aiohttp_session.post(url, data={'context': query_text}) as response:
            result = await response.json()
            if result['success']:
                results = result['results']
                for idx in range(len(results)):
                    results[idx]['cdn_img'] = IMAGE_CDN+results[idx]['image_hash']
                return jinja.render("img_results.html", request, results=results, text=query_text, text_url=text_url)
    return redirect('/')

@app.route("/", methods=["POST","GET"])
async def index(request):
    if request.method == 'POST':
        if 'query' in request.form:
            query_text = request.form.get('query')
            url = app.url_for('search_text', q=query_text)
            return redirect(url)
        if 'image' in request.files:
            image_raw = request.files['image'][0].body
            files = {'image': image_raw}
            url = IMG_ML_DOMAIN+ENCODED_IMG
            async with app.aiohttp_session.post(url, data=files) as response:
                result = await response.json()
            img_encoding = result['encoding']
            str_img_encoding = ','.join([ '{:.3f}'.format(f) for f in img_encoding])
            url = app.url_for('search_image', v=str_img_encoding)
            return redirect(url)

        if 'image_query' in request.form:
            query_text = request.form.get('image_query')
            url = app.url_for('search_image', q=query_text)
            return redirect(url)

    return jinja.render("landing.html", request)

@app.route("/privacy")
async def show_privacy(request):
    return jinja.render("privacy.html", request)

@app.route("/terms")
async def show_terms(request):
    return jinja.render("terms.html", request)
