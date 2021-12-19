from aiohttp import web
from sourse import GensimLsi, GensimVectorizer, TextNormalizer

import json
import pickle
routes = web.RouteTableDef()

clf = pickle.load(open('model.pkl', 'rb'))


@routes.post('/api/text')
async def create_report(request):
    data = await request.post()
    print(data['text'])
    text = data['text']
    res = clf.predict([text])
    print(res[0])
    return web.Response(text=json.dumps({'text': res[0]}), status=200)

app = web.Application()
app.add_routes(routes)
web.run_app(app)
