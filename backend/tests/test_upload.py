import requests



if __name__ == "__main__":
    # img_file = '../univec/8108923536.jpg'
    # url = 'http://127.0.0.1:5005/inference/img/encode/img'
    # files = {'image': open(img_file, 'rb')}
    # res = requests.post(url, files=files)
    # print(res.json()['encoding'][:10])


    url = 'http://127.0.0.1:5005/inference/img/end2end/img'
    #res = requests.post(url, files=files)
    #print(res.json()['encoding'][:10])

    url = 'http://127.0.0.1:5005/inference/text/end2end/text'
    res = requests.post(url, data={'context': "This is a text"})
    # print(res.json()['encoding'][:10])

    # url = 'http://127.0.0.1:5005/inference/text/encode/text'
    # res = requests.post(url, data={'context': "This is a text"})
    # print(res.json()['encoding'][:10])
