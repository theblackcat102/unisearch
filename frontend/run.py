from universe.app import app
from universe.settings import DEBUG 

if __name__ == '__main__':
	if DEBUG:
		HOST = '127.0.0.1'
		from aoiklivereload import LiveReloader
		reloader = LiveReloader()
		reloader.start_watcher_thread()
		app.run(host=HOST, port=8000, debug=DEBUG, access_log=DEBUG)
	else:
		HOST = '0.0.0.0'
		app.run()
	#app.run(host=HOST, port=5000, debug=DEBUG, access_log=DEBUG)
