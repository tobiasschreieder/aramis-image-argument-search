import datetime
import logging
import os
from pathlib import Path

from flask import Flask, render_template, request, send_file, abort, jsonify, make_response

from indexing import DataEntry, Topic
from retrieval import RetrievalSystem
from evaluation import save_eval, Stance, Argumentative, get_image_to_eval

log = logging.getLogger('frontend.flask')
app = Flask(__name__, static_url_path='', static_folder='static')

retrieval_system: RetrievalSystem = None
image_ids = DataEntry.get_image_ids()


def start_server(system: RetrievalSystem, debug: bool = True, host: str = '0.0.0.0', port: int = 5000):
    log.debug('starting Flask')
    app.secret_key = '977e39540e424831d8731b8bf17f2484'
    app.debug = debug
    global retrieval_system
    retrieval_system = system
    app.run(host=host, port=port, use_reloader=False)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if retrieval_system is not None:
            if 'query' in request.form.keys() and 'topK' in request.form.keys():
                then = datetime.datetime.now()
                try:
                    top_k = int(request.form['topK'])
                except ValueError:
                    top_k = 20
                pro_result, con_result = retrieval_system.query(request.form['query'], top_k=top_k)
                now = datetime.datetime.now()
                pro_images = [DataEntry.load(iid[0]) for iid in pro_result]
                con_images = [DataEntry.load(iid[0]) for iid in con_result]
                return render_template('index.html',
                                       pros=pro_images, cons=con_images,
                                       search_value=request.form['query'], topK=top_k,
                                       time_request=str(now-then))

    return render_template('index.html', pros=[], cons=[], topK=20)


@app.route('/evaluation', methods=['GET', 'POST'])
def evaluation():
    user = request.cookies.get('user_name', '')
    topics = Topic.load_all()
    selected_topic = Topic.get(1)
    image = None

    topic_done = False

    if request.method == 'POST':
        if 'selected_topic' in request.form.keys() and 'user_name' in request.form.keys():
            user = request.form['user_name']
            try:
                selected_topic = Topic.get(int(request.form['selected_topic']))
            except ValueError:
                pass

        if len(user) > 0:
            if 'arg' in request.form.keys() and 'stance' in request.form.keys() \
                    and 'image_id' in request.form.keys() and 'topic_correct' in request.form.keys():
                if request.form['arg'] == 'weak':
                    arg = Argumentative.WEAK
                elif request.form['arg'] == 'strong':
                    arg = Argumentative.STRONG
                else:
                    arg = Argumentative.NONE

                if request.form['stance'] == 'pro':
                    stance = Stance.PRO
                elif request.form['stance'] == 'con':
                    stance = Stance.CON
                else:
                    stance = Stance.NEUTRAL

                if request.form['topic_correct'] == 'topic-true':
                    topic_correct = True
                else:
                    topic_correct = False

                save_eval(image_id=request.form['image_id'], user=user, topic_correct=topic_correct,
                          topic=selected_topic.number, arg=arg, stance=stance)

    if len(user) > 0:
        image = get_image_to_eval(selected_topic)
        if image is None:
            topic_done = True

    resp = make_response(render_template('evaluation.html', topics=topics, selected_topic=selected_topic,
                                         user_name=user, topic_done=topic_done, image=image))
    expire = datetime.datetime.now() + datetime.timedelta(days=90)
    if len(user) > 0:
        resp.set_cookie('user_name', user, expires=expire)
    return resp


def get_abs_data_path(path):
    if not path.is_absolute():
        path = Path(os.path.abspath(__file__)).parent.parent.joinpath(path)
    return path


@app.route('/data/image/<path:image_id>')
def data_image(image_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    return send_file(get_abs_data_path(entry.png_path))


@app.route('/data/screenshot/<path:image_id>/<path:page_id>')
def data_snp_screenshot(image_id, page_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    for page in entry.pages:
        if page.url_hash == page_id:
            return send_file(get_abs_data_path(page.snp_screenshot))
    return abort(404)


@app.route('/data/dom/<path:image_id>/<path:page_id>')
def data_snp_dom(image_id, page_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    for page in entry.pages:
        if page.url_hash == page_id:
            return send_file(get_abs_data_path(page.snp_dom))
    return abort(404)
