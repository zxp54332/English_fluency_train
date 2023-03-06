# import re
import base64
import fcntl
import json
import math
import os
import sys
import tempfile
import time
import traceback
import urllib

import ffmpy
import numpy as np
import requests
import soundfile as sf
from flask_apiexceptions import ApiError, ApiException
from g2p_en import G2p
from nltk.tokenize import TweetTokenizer
from ponddySyllableBreaking import getSyllableBreakingSingleUtt

# from g2p import G2p
from pydub import AudioSegment

word_tokenize = TweetTokenizer().tokenize

JWT_PRODUCTION_SERVER = "https://api.ponddy.com"
JWT_PRODUCTION = "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6InBvbmRkeS1zdXBlciIsImV4cCI6MTU0ODEyODM4OSwiZW1haWwiOiIifQ.WK06nkIlY0Fo51vy2pajtP_K1g2-YDvZTPFFsq_m44I"
JWT_STAGING_SERVER = "https://api-staging.ponddy.com"
JWT_STAGING = "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6InBvbmRkeS1zdXBlciIsImV4cCI6MTU2ODE4NDg3MiwiZW1haWwiOiIifQ.U-WtxF6jzGpTE4hWvRp8umjvCguvCJG7HdUxyToD-7U"
JWT_DEVELOP_SERVER = "https://api-dev.ponddy.com"
JWT_DEVELOP = "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6InBvbmRkeS1zdXBlciIsImV4cCI6MTUzNTY5NTExNywiZW1haWwiOiIifQ.A6KqSO4JNMuH8r_qEi447_xMEccg7QjiPircLSF7GS8"
JWT_SJ = "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6InBvbmRkeSIsImV4cCI6MTU4NDcyNzI2NywiZW1haWwiOiJmcmFuY29jaGVuQHBvbmRkeS1lZHUuY29tIn0.DZArlDZwi5q4H3CxAilKj0-XX2HOhOTMJNrniLaImC4"
# JWT_SJ_SERVER = 'http://sj-server3.ponddy-one.com'
JWT_SJ_SERVER = "http://sj-server2.ponddy-one.com"
# taipei ML server
JWT_TAIPEIML_SERVER = "http://alpha.ponddy-one.com:8800"
JWT_TAIPEIML = "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFkbWluIiwiZXhwIjoxNTU3MTI1NTQ4LCJlbWFpbCI6Im93ZW56aG9uZ0Bwb25kZHktZWR1LmNvbSJ9.6YB9Kvy2ZDUv4L16NYibpAiQTjMR2ug-WU6B28m9g4E"

minscores_angel = 60
minscores_rigorous = 80
minscores_missing = 15
low_pass_width = 6000

chi_puncts = "﹐﹑，。？、：；⋯:;,?!.！「」（）()】【：:╱〈〉{}[]“”《》"
punctuations = set('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~' + chi_puncts)
LOGFILEPATH = "./access_parameter.log"
# ENGLISH_WEB_ROOT = 'https://85.ponddy-one.com:9479'
# ENGLISH_WEB_ROOT = os.environ.get('ENGLISH_WEB_ROOT', 'https://sj-server3.ponddy-one.com:9479')
# ENGLISH_WEB_ROOT = os.environ.get('ENGLISH_WEB_ROOT', 'https://sj-server2.ponddy-one.com:9479')
ENGLISH_WEB_ROOT = os.environ.get("ENGLISH_WEB_ROOT", "https://192.168.1.27:9479")

Widget_level_cache = "./Widget_level_cache.npy"
word2level = {}
if os.path.isfile(Widget_level_cache):
    print("load...level cache")
    word2level = np.load(Widget_level_cache, allow_pickle=True).item()

SELECT_SERVER = os.environ.get("SELECT_SERVER", "develop")

# 處理setence 存取 setence等級的快取
def get_level_cache_status(inputText, forcesave=False):
    speakwords_encode = urllib.parse.quote(inputText, safe="")
    cache_lingo_key = "level__%s" % speakwords_encode
    if word2level.get(cache_lingo_key, ""):
        # print('++++++Read from cache')
        inputText, input_text_lv = word2level[cache_lingo_key]
    else:
        # {'text': 'He makes more money than his dad and uncle.', 'prediction': 'A2', 'prediction_list': 'A2, C1, A1'}
        retdic = get_sentence_level(inputText)
        input_text_lv = retdic.get("prediction")
        if input_text_lv:
            word2level[cache_lingo_key] = [inputText, input_text_lv]

    if int(time.time()) % 3 == 1 or forcesave:
        # print('%%%%%%Save pinyin cache:%s' % Widget_level_cache)
        file = open(Widget_level_cache, "a+b")
        fcntl.flock(file, fcntl.LOCK_EX)
        np.save(Widget_level_cache, word2level)
        fcntl.flock(file, fcntl.LOCK_UN)
        file.close()
    return inputText, input_text_lv


def toCMUSeq(sent):
    print("toCMUSeq", sent)
    g2p = G2p()
    res = g2p(sent)
    cmu = res[0]
    tmp = []
    buf = []
    for c in cmu:
        if c == " " or c in punctuations:
            if (
                len(tmp) > 0
                and tmp[-1] == "DH_AH0"
                and any(buf[0].startswith(x) for x in "AEIOU")
            ):
                tmp[-1] = "DH_IY1"
            if len(buf) > 0:
                tmp.append("_".join(buf))
            buf = []
        else:
            buf.append(c)
    if len(buf) > 0:
        if (
            len(tmp) > 0
            and tmp[-1] == "DH_AH0"
            and any(buf[0].startswith(x) for x in "AEIOU")
        ):
            tmp[-1] = "DH_IY1"
        tmp.append("_".join(buf))
    return " ".join(tmp), res[1]


def word_score_offset(gopresult):
    # offset = 20
    targets = "I, his, the, a, an, and, are".upper().split(", ")
    for w in gopresult:
        if w["word"] in targets:
            w["GOPScore"] = min(100, w["GOPScore"] + 20)
    return gopresult


# Apply low pass filter
def sr_change(inputFilePath, outputFilePath, LPF=8000, SR=16000):
    ff = ffmpy.FFmpeg(
        inputs={inputFilePath: None},
        outputs={
            outputFilePath: '-af "lowpass=f=' + str(LPF) + '" -ar ' + str(SR) + " -y"
        },
    )
    ff.run()
    if not os.path.isfile(outputFilePath):
        print("Error in converting sample rate for file %s" % (outputFilePath))


# 保留使用者記錄，當wrtdata格式有base64資料時，會將檔案儲存於 ./logs/{hashkey_uuid}.base64 方便存取及讀取資料
def savelog(hashkey_uuid, action, wrtdata):
    # print('----savelog----')
    file = open(LOGFILEPATH, "a+")
    fcntl.flock(file, fcntl.LOCK_EX)
    tm = time.localtime()
    date = "%04d/%02d/%02d" % (tm[0], tm[1], tm[2])
    wrtdatalog = ""
    if type(wrtdata) == dict:
        wrtdatalog = wrtdata.copy()
        if wrtdatalog.get("base64", "") and hashkey_uuid:
            wrtbasebs = "./logs"
            if not os.path.isdir(wrtbasebs):
                os.mkdir(wrtbasebs)
            wrtbasedir = "%s/%s" % (wrtbasebs, hashkey_uuid[0:6])
            if not os.path.isdir(wrtbasedir):
                os.mkdir(wrtbasedir)
            wrtbasepath = "%s/%s.base64" % (wrtbasedir, hashkey_uuid)
            if not os.path.isfile(wrtbasepath):
                wf = open(wrtbasepath, "w")
                wf.write("%s" % wrtdatalog.get("base64", ""))
                wf.close()
            del wrtdatalog["base64"]

    file.write("=========[%s]%s(%s)=========\n" % (date, hashkey_uuid, action))
    if wrtdatalog:
        file.write("%s\n" % wrtdatalog)
    else:
        file.write("%s\n" % wrtdata)
    fcntl.flock(file, fcntl.LOCK_UN)
    file.close()


def getEngASRprediction(wav_path):
    """Get results from ponddy English ASR api.

    Args:
      wav_path: the path of the input waveform file

    Returns:
      result: ASR prediction results
    """
    enc = base64.b64encode(open(wav_path, "rb").read())
    enc = enc.decode("utf-8")
    data = {"base64": enc}
    r = requests.post(
        "http://api.ponddy.com/api/voice_asr/en/predict",
        json=data,
        headers={
            "Authorization": "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6InBvbmRkeS1zdXBlciIsImV4cCI6MTU5MDA5MTI4NiwiZW1haWwiOiIifQ.d3IPqFIK8RP8FX7d5dYDRMW2iHol6JEysncOc8NoOVo"
        },
    )
    r.encoding = "utf-8"
    result = json.loads(r.text)
    return result


def getEngGOPresult(wav_path, text, client, cmu_ans=None, accountid="", logstatus=True):
    """Get results from ponddy English GOP api.

    Args:
      wav_path: the path of the input waveform file
      text: input text as ground truth

    Returns:
      result: GOP scoring results
    """
    enc = base64.b64encode(open(wav_path, "rb").read())
    enc = enc.decode("utf-8")
    data = {
        "base64": enc,
        "words": text,
        "appname": "widget",
        "diagnosis_by": "GOP_Result",
        "dereverb_check": "N",
    }

    # wf = open('/tmp/err_english', 'w')
    # wf.write('%s' % data)
    # wf.close()

    if cmu_ans is not None:
        data["cmu_ans"] = cmu_ans
    if accountid:
        data["uuid"] = accountid
    # print('logstatus', logstatus)
    if not logstatus:
        data["logsave"] = "N"
    select_server_type = "(getEngGOPresult)"
    # production / staging / develop / SJ
    select_server = SELECT_SERVER
    # select_server = 'taipeiml'
    print("select_server", select_server)
    if select_server == "production":
        print("===production%s===" % select_server_type)
        server_url = JWT_PRODUCTION_SERVER
        headers = {"Authorization": JWT_PRODUCTION}
    elif select_server == "staging":
        print("===staging%s===" % select_server_type)
        server_url = JWT_STAGING_SERVER
        headers = {"Authorization": JWT_STAGING}
    elif select_server == "develop":
        print("===develop%s===" % select_server_type)
        server_url = JWT_DEVELOP_SERVER
        headers = {"Authorization": JWT_DEVELOP}
    elif select_server == "taipeiml":
        server_url = JWT_TAIPEIML_SERVER
        headers = {"Authorization": JWT_TAIPEIML}
    elif select_server == "SJ":
        print("===SJ%s===" % select_server_type)
        server_url = JWT_SJ_SERVER
        headers = {"Authorization": JWT_SJ}
    r = client.post(
        "%s/api/voice_score/en/diagnosis" % server_url, json=data, headers=headers
    )
    r.encoding = "utf-8"
    # print("r.status_code", r.status_code)
    result = json.loads(r.text)

    if r.status_code == 422:
        # result {'detail': 'The input word must be chinese!', 'code_sn': 'c10'}
        code_sn = result.get("code_sn")
        code_message = result.get("detail")
        error_silent = ApiError(code=code_sn, message=code_message)
        raise ApiException(status_code=422, error=error_silent)
    # print('result', result)
    return result


def getEngTTS(text):
    """Get results from ponddy English TTS api.

    Args:
      text: input text for speech synthesis

    Returns:
      result: generated audio with base64 encoded and other information
    """
    select_server_type = "(EngTTS)"
    # production / staging / develop / SJ
    # select_server = 'production'
    select_server = SELECT_SERVER
    if select_server == "production":
        print("===production%s===" % select_server_type)
        server_url = JWT_PRODUCTION_SERVER
        headers = {"Authorization": JWT_PRODUCTION}
    elif select_server == "staging":
        print("===staging%s===" % select_server_type)
        server_url = JWT_STAGING_SERVER
        headers = {"Authorization": JWT_STAGING}
    elif select_server == "develop":
        print("===develop%s===" % select_server_type)
        server_url = JWT_DEVELOP_SERVER
        headers = {"Authorization": JWT_DEVELOP}
    elif select_server == "SJ":
        print("===SJ%s===" % select_server_type)
        server_url = JWT_SJ_SERVER
        headers = {"Authorization": JWT_SJ}
    if select_server == "SJ":
        data = {"text": text, "gender": "male"}
        r = requests.post("%s:8508/" % JWT_SJ_SERVER, json=data)
    else:
        data = {"text": text, "gender": "female", "datatype": "mp3"}
        r = requests.post(
            "%s/api/voice_tts/en/predict" % server_url, json=data, headers=headers
        )
        # print('r', r.text)
    # r = requests.post('http://api.ponddy.com/api/voice_tts/en/predict', json=data, headers={'Authorization': 'JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6InBvbmRkeS1zdXBlciIsImV4cCI6MTU5MDA5MTI4NiwiZW1haWwiOiIifQ.d3IPqFIK8RP8FX7d5dYDRMW2iHol6JEysncOc8NoOVo'})
    r.encoding = "utf-8"
    result = json.loads(r.text)
    return result


def getEngRecommendSents(gop_dict):
    data = {"uttGOP": gop_dict}
    print("data", data)
    try:
        r = requests.post(
            "%s:6954/eng_sent_recommend" % JWT_SJ_SERVER,
            json=data,
            headers={"Connection": "close"},
        )
        # print('result', result)
        result = json.loads(r.text)
        return result
    except Exception:
        traceback.print_exc()
        sys.exit()


def get_sentence_level(ss):
    data = {"text": ss}
    select_server_type = "(Sentence_level)"
    # production / staging / develop / SJ
    select_server = SELECT_SERVER
    if select_server == "production":
        print("===production%s===" % select_server_type)
        server_url = JWT_PRODUCTION_SERVER
        headers = {"Authorization": JWT_PRODUCTION}
    elif select_server == "staging":
        print("===staging%s===" % select_server_type)
        server_url = JWT_STAGING_SERVER
        headers = {"Authorization": JWT_STAGING}
    elif select_server == "develop":
        print("===develop%s===" % select_server_type)
        server_url = JWT_DEVELOP_SERVER
        headers = {"Authorization": JWT_DEVELOP}
    elif select_server == "SJ":
        print("===SJ%s===" % select_server_type)
        server_url = JWT_SJ_SERVER
        headers = {"Authorization": JWT_SJ}
    r = requests.post(
        "%s/api/sentence_level/en/predict" % server_url, json=data, headers=headers
    )
    r.encoding = "utf-8"
    # print("r.status_code", r.status_code)
    if r.status_code == 200:
        result = json.loads(r.text)
    else:
        result = {}
    return result


def getFluencyLevel(wav_path):
    """Get results of fluency level model

    Args:
      wav_path: the path of the input waveform file

    Returns:
      result: fluency level results
    """
    enc = base64.b64encode(open(wav_path, "rb").read())
    enc = enc.decode("utf-8")
    data = {"base64": enc}
    select_server_type = "(Fluency)"
    # production / staging / develop / SJ
    # 2020.09.11 develop english fluency 已經可以運作了
    select_server = SELECT_SERVER
    # print('data', data)
    if select_server == "production":
        print("===production%s===" % select_server_type)
        server_url = JWT_PRODUCTION_SERVER
        headers = {"Authorization": JWT_PRODUCTION}
    elif select_server == "staging":
        print("===staging%s===" % select_server_type)
        server_url = JWT_STAGING_SERVER
        headers = {"Authorization": JWT_STAGING}
    elif select_server == "develop":
        print("===develop%s===" % select_server_type)
        server_url = JWT_DEVELOP_SERVER
        headers = {"Authorization": JWT_DEVELOP}
    elif select_server == "SJ":
        print("===SJ%s===" % select_server_type)
        server_url = JWT_SJ_SERVER
        headers = {"Authorization": JWT_SJ}
    if select_server == "SJ":
        r = requests.post("%s:3018/fluency_score" % JWT_SJ_SERVER, json=data)
    else:
        r = requests.post(
            "%s/api/voice_fluency/en/predict" % server_url, json=data, headers=headers
        )
    result = json.loads(r.text)
    # print('FFFFFFFFFFFFFFFFFFFFF', result)
    return result


def CEFR_2_level(level):
    cefrlevel = ""
    tocflevel = ""
    if level == "Pre A1":
        cefrlevel = "Pre A1"
        tocflevel = "Pre A1"
    elif level == "A1":
        cefrlevel = "A1"
        tocflevel = "A1"
    elif level == "A2":
        cefrlevel = "A2"
        tocflevel = "A2"
    elif level == "B1":
        cefrlevel = "B1"
        tocflevel = "B1"
    elif level == "B2":
        cefrlevel = "B2"
        tocflevel = "B2"
    elif level == "C1":
        cefrlevel = "C1"
        tocflevel = "C1"
    elif level == "C2":
        cefrlevel = "C2"
        tocflevel = "C1"
    return cefrlevel, tocflevel


# {'5': '#C638AD', '0': '#71E67C', '6': '#E54956', '2': '#8BC7EF', '4': '#511AE8', '3': '#3E86F9', '1': '#02CCC2',
#  'U': '#BABABA'}
def CEFR_2_color(level):
    colorv = ""
    # 0
    if level == "Pre A1":
        colorv = "#71E67C"
    # 1
    elif level == "A1":
        colorv = "#02CCC2"
    # 2
    elif level == "A2":
        colorv = "#8BC7EF"
    # 3
    elif level == "B1":
        colorv = "#3E86F9"
    # 4
    elif level == "B2":
        colorv = "#511AE8"
    # 5
    elif level == "C1":
        colorv = "#C638AD"
    # 6
    elif level == "C2":
        colorv = "#E54956"
    return colorv


def getSentSet(maxlength=99):
    f = open("data/sent_set.txt")
    lines = f.readlines()
    retlist = []

    sentences_structure = []
    # no_show_num_pat = re.compile(r'[0-9]+')
    for line in lines:
        line = line.strip()
        #         if len(line) <= maxlength and not re.search(no_show_num_pat, line):
        #             if line not in retlist:
        #                 retlist.append(line)
        #                 ret_text, ret_text_lv = get_level_cache_status(line)
        #                 # print(ret_text, ret_text_lv)
        #                 cefr_color = CEFR_2_color(ret_text_lv)
        #                 sentences_structure.append({'en': line, 'level': {'CEFR': ret_text_lv, 'CEFR_color': cefr_color}})
        sent, lv = line.split("\t")
        if sent not in retlist:
            retlist.append(sent)
            cefr_color = CEFR_2_color(lv)
            sentences_structure.append(
                {"en": sent, "level": {"CEFR": lv, "CEFR_color": cefr_color}}
            )
    # lines = [line.strip() for line in lines]
    f.close()
    # print('sentences_structure', sentences_structure)
    return retlist, sentences_structure


def wer(ref, hyp, debug=False):
    """Calculate edit distance between two input sequences and return the results.

    Args:
      ref: the ground truth sequence as reference
      hyp: the prediction results as hypothesis

    Returns:
      result_dict: the dictionary contains all informations
    """
    SUB_PENALTY = 1
    INS_PENALTY = 1
    DEL_PENALTY = 1
    r = ref.split()
    h = hyp.split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = (
                    costs[i - 1][j - 1] + SUB_PENALTY
                )  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
    lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            lines.append("DEL\t" + r[i] + "\t" + "****")
    tmp = [x.split() for x in lines]
    tmp = list(map(list, zip(*tmp)))
    info = [x[::-1] for x in tmp]
    order = [1, 2, 0]
    info = [info[i] for i in order]
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    wer_result = round((numSub + numDel + numIns) / (float)(len(r)), 3)
    return {
        "WER": wer_result,
        "Cor": numCor,
        "Sub": numSub,
        "Ins": numIns,
        "Del": numDel,
        "word_count": len(r),
        "total_err": (numSub + numDel + numIns),
        "info": info,
    }


def generateEditDistanceResult(wav_path, inputText):
    """Wrapped function to generate edit distance results from input waveform file and input text.
       Apply ASR first to get ASR prediction as hypothesis and then apply wer function to calculate
       edit distance between ASR prediction and input text.

    Args:
      wav_path: the path of the input waveform file
      inputText: input text

    Returns:
      ref: the list of all words in reference text
      hyp: the list of all words in hypothesis text
      ins_indices: the indices of all insertions
      del_indices: the indices of all deletions
      sub_indices: the indices of all substitutions
    """
    inputText = inputText.upper()
    puncts_dict = {}
    alternated_text = []
    for i, wd in enumerate(inputText.split()):
        if wd in punctuations:
            puncts_dict[i] = wd
        else:
            alternated_text.append(wd)
    inputText = " ".join(alternated_text)

    res = getEngASRprediction(wav_path)

    predicted_text = res["asr"]
    # print('ASR', predicted_text)
    error_silent = ApiError(
        code="Unable to analyze", message="Unable to analyze, please try again."
    )
    if predicted_text.strip() == "":
        raise ApiException(status_code=422, error=error_silent)

    result = wer(ref=inputText, hyp=predicted_text, debug=False)
    ref, hyp, label = result["info"]

    puncts_offset = [0] * len(puncts_dict)
    for i, wd in enumerate(hyp):
        if ref[i] == "****":
            for j, idx in enumerate(puncts_dict.keys()):
                if idx + puncts_offset[j] >= i:
                    puncts_offset[j] += 1
        if not label[i] == "INS":
            hyp[i] = ref[i]
    # print('puncts_dict', puncts_dict)
    # for idx in puncts_dict:
    for i, idx in enumerate(puncts_dict):
        ref.insert(idx + puncts_offset[i], puncts_dict[idx])
        hyp.insert(idx + puncts_offset[i], puncts_dict[idx])
        label.insert(idx + puncts_offset[i], puncts_dict[idx])

    ins_indices, del_indices, sub_indices = [], [], []
    for i, x in enumerate(label):
        if x == "INS":
            ins_indices.append(i)
        elif x == "SUB":
            sub_indices.append(i)
        elif x == "DEL":
            del_indices.append(i)
    # ins_indices = [i for i,x in enumerate(label) if x=="INS"]
    # del_indices = [i for i,x in enumerate(label) if x=="DEL"]
    # sub_indices = [i for i,x in enumerate(label) if x=="SUB"]
    acc = (1 - result["WER"]) * 100
    return ref, hyp, ins_indices, del_indices, sub_indices, acc


def generatePartialGOPResult(
    wav_path, inputText, hashkey_uuid="", logstatus=True, prtmsg=False
):
    """Wrapped function to generate partial GOP results from input waveform file and input text..

    Args:
      wav_path: the path of the input waveform file
      inputText: input text

    Returns:
      ref: the list of all words in reference text
      hyp: the list of all words in hypothesis text
      ins_indices: the indices of all insertions
      del_indices: the indices of all deletions
      sub_indices: the indices of all substitutions
    """
    cmu_ans, inputText = toCMUSeq(inputText)
    inputText = inputText.upper()
    puncts_dict = {}
    alternated_text = []
    for i, wd in enumerate(inputText.split()):
        if wd in punctuations:
            puncts_dict[i] = wd
        else:
            alternated_text.append(wd)
    inputText = " ".join(alternated_text)
    if prtmsg:
        print("GOP input text:%s" % inputText)
    print("logstatus - 2", logstatus)
    res = getEngGOPresult(
        wav_path=wav_path, text=inputText, cmu_ans=cmu_ans, logstatus=logstatus
    )
    if prtmsg:
        print("str(res)", str(res)[0:200])
    error_silent = ApiError(
        code="e02", message="Unable to analyze, Listen once more and try again."
    )
    gopresult = res.get("gop", {}).get("parts", "")
    if gopresult == "":
        savelog(
            hashkey_uuid, "diagnosis:ERROR", "Gop Unable to analyze, please try again."
        )
        raise ApiException(status_code=422, error=error_silent)

    res["gop"]["parts"] = word_score_offset(res["gop"]["parts"])
    gop_results = [
        (x["GOPScore"] >= minscores_angel, x["GOPScore"])
        for _, x in enumerate(res["gop"]["parts"])
    ]

    ipas_dicts = getSyllableBreakingSingleUtt(res["gop"])
    ipa_ans = [" ".join(d["ipa_break_merged"]) for d in ipas_dicts]

    # print('ipas_dicts', ipas_dicts)
    ipa_pred = []
    for i, d in enumerate(ipas_dicts):
        ipa_pred_tmp = []
        for dv in d["ipa_break_pred_merged"]:
            if type(dv) == str:
                ipa_pred_tmp.append(dv)
            elif type(dv) == list:
                ipa_pred_tmp.append("".join(dv))
        cur_pred_ipa = " ".join(ipa_pred_tmp)
        # 如果預測 IPA 與答案 IPA 相符，即使 gop 分數低於70，都需要強迫設定成70分
        # 如果預測 IPA 與答案 IPA 相符，即使 gop 分數低於80，都需要強迫設定成80分
        if cur_pred_ipa == ipa_ans[i] and gop_results[i][1] < 80:
            gop_results[i] = (True, 80)
        ipa_pred.append(cur_pred_ipa)

    predicted_text = " ".join([x["word"] for _, x in enumerate(res["gop"]["parts"])])
    result = wer(ref=inputText, hyp=predicted_text, debug=False)
    ref = word_tokenize(inputText)
    _, hyp, label = result["info"]
    puncts_offset = [0] * len(puncts_dict)
    for i, wd in enumerate(hyp):
        if ref[i] == "****":
            for j, idx in enumerate(puncts_dict.keys()):
                if idx + puncts_offset[j] >= i:
                    puncts_offset[j] += 1
        if not label[i] == "INS":
            hyp[i] = ref[i]

    acc = np.mean([x[1] for x in gop_results])
    # print('puncts_dict', puncts_dict)
    # for idx in puncts_dict:
    for i, idx in enumerate(puncts_dict):
        ref.insert(idx + puncts_offset[i], puncts_dict[idx])
        hyp.insert(idx + puncts_offset[i], puncts_dict[idx])
        label.insert(idx + puncts_offset[i], puncts_dict[idx])
        gop_results.insert(idx + puncts_offset[i], (True, puncts_dict[idx]))

    ins_indices, del_indices, sub_indices = [], [], []
    for i, x in enumerate(label):
        if x == "INS":
            ins_indices.append(i)
        elif x == "SUB":
            sub_indices.append(i)
        elif x == "DEL":
            del_indices.append(i)
    # ins_indices = [i for i,x in enumerate(label) if x=="INS"]
    # del_indices = [i for i,x in enumerate(label) if x=="DEL"]
    # sub_indices = [i for i,x in enumerate(label) if x=="SUB"]
    #     del_indices, sub_indices = [], []
    #     for i, x in enumerate(gop_results):
    #         try:
    #             if x[1] < minscores_missing:
    #                 del_indices.append(i)
    #             elif minscores_missing<=x[1]<minscores_angel:
    #                 sub_indices.append(i)
    #         except:
    #             pass

    #     del_indices = [i for i,x in enumerate(gop_results) if not x[0] and x[1]<minscores_missing]
    sub_indices = [i for i, x in enumerate(gop_results) if not x[0]]
    # acc = (1-result['WER'])*100
    #     acc = (1-(len(sub_indices)/len(gop_results)))*100
    return ref, hyp, ins_indices, del_indices, sub_indices, acc


def generateGOPResult(
    wav_path,
    asr_lvs_score,
    inputText,
    ref,
    hyp,
    ins_indices,
    del_indices,
    sub_indices,
    hashkey_uuid,
    accountID="",
    logstatus=True,
    prtmsg=False,
):
    """Wrapped function to generate gop results.
       Modify the input text according to the indices of insertions, deletions and substitutions first
       to avoid gop misaligned error.

    Args:
      wav_path: the path of the input waveform file
      ref: the list of all words in reference text
      hyp: the list of all words in hypothesis text
      ins_indices: the indices of all insertions
      del_indices: the indices of all deletions
      sub_indices: the indices of all substitutions

    Returns:
      gop_results: the results of gop scoring
      ipas: syllable based ipa of the text
    """
    if prtmsg:
        print("======GOP result{S}======")
        print("hyp", hyp)
        print("ref", ref)
        print("ins_indices(多)", ins_indices)
        print("del_indices(少)", del_indices)
        print("ins_indices(錯)", ins_indices)
    puncts_indices = []
    inputText = " ".join(inputText)
    # print('DEBUG: inputText:', inputText)
    cmu_ans, _ = toCMUSeq(inputText)
    # print('DEBUG: cmu_ans = ', cmu_ans)
    alternated_text = []
    for i, wd in enumerate(hyp):
        if wd in punctuations:
            puncts_indices.append(i)
        else:
            alternated_text.append(wd)
    text = " ".join(alternated_text)

    res = getEngGOPresult(
        wav_path=wav_path,
        text=text,
        cmu_ans=cmu_ans,
        accountid=accountID,
        logstatus=logstatus,
    )
    if prtmsg:
        print("GOP text", text)
        print("puncts_indices", puncts_indices)
        print("GOP response json", str(res)[0:150])

    error_silent = ApiError(
        code="e02", message="Unable to analyze, Listen once more and try again."
    )
    gopresult = res.get("gop", {}).get("parts", "")
    if gopresult == "":
        savelog(
            hashkey_uuid, "diagnosis:ERROR", "Gop Unable to analyze, please try again."
        )
        raise ApiException(status_code=422, error=error_silent)

    detail_parts = res["gop"]["parts"]
    # print('detail_parts', detail_parts)
    parts_ctm_lists = do_userwav_ctm(wav_path, detail_parts)
    minscores_angel = 60
    res["gop"]["parts"] = word_score_offset(res["gop"]["parts"])
    gop_results = [
        (
            x["GOPScore"] >= minscores_angel,
            x["GOPScore"],
            x.get("word_model_active", ""),
        )
        for _, x in enumerate(res["gop"]["parts"])
    ]
    ipas_dicts = getSyllableBreakingSingleUtt(res["gop"])
    ipa_ans = [" ".join(d["ipa_break_merged"]) for d in ipas_dicts]

    print("gop_results--1", gop_results)
    # print('ipas_dicts', ipas_dicts)
    ipa_pred = []
    for i, d in enumerate(ipas_dicts):
        ipa_pred_tmp = []
        for dv in d["ipa_break_pred_merged"]:
            if type(dv) == str:
                ipa_pred_tmp.append(dv)
            elif type(dv) == list:
                ipa_pred_tmp.append("".join(dv))
        cur_pred_ipa = " ".join(ipa_pred_tmp)
        # 如果預測 IPA 與答案 IPA 相符，即使 gop 分數低於80，都需要強迫設定成80分
        if cur_pred_ipa == ipa_ans[i] and gop_results[i][1] < 80:
            gop_results[i] = (True, 80, gop_results[i][2])
        ipa_pred.append(cur_pred_ipa)
    print("gop_results--2", gop_results)
    # 如果GOP分數有達C級，則讓 ipa_pred[i] = ipa_ans[i]
    for i, (gs, gv, word_model) in enumerate(gop_results):
        if gv >= 80:
            ipa_pred[i] = ipa_ans[i]

    print("ipa_pred", ipa_pred)
    print("gop_results", gop_results)
    # ['ɪt', '*ʌv', 'faɪŋ', 'naʊ']
    # ipa_pred = [' '.join(d['ipa_break_pred_merged']) for d in ipas_dicts]
    # print('===ipa_pred===', ipa_pred)

    # 計算gop 分數 * Levenshtein(asr)的分數
    gop_correct = 0
    for gr, gs, word_model in gop_results:
        if str(gr).lower() == "true":
            gop_correct += 1
    gop_score = gop_correct / len(gop_results)
    gop_score = round(gop_score * 100)
    if prtmsg:
        print("gop_results", gop_results)
        print("asr_lvs_score", asr_lvs_score)
        print("ipa_ans", ipa_ans)
        print("ipa_pred", ipa_pred)
        print("gop_score", gop_score)
    # gop_score_lvs = round((gop_score * asr_lvs_score) / 100)
    gop_score_lvs = round(np.mean([x[1] for x in gop_results]))
    # print('gop_score_lvs', gop_score_lvs)

    # 星等分數
    total_star = round(gop_score_lvs / 20, 1)
    # print('total_star', total_star)

    # recommended_sents_res = getEngRecommendSents(res['gop'])
    recommended_sents_res = {}
    recommended_sents_res["recommendation"] = [{"recommend": []}]
    # print("recommended_sents_res['recommendation']", recommended_sents_res['recommendation'])
    # TTS 挑選100字以下
    recommended_sentences = []

    # [(False, 35.50183903770348), (True, 73.59730599132585), (False, 12.28575122817972), (True, 100), (False, 14.015328757050762)]
    diagnosis_error_index = []
    for i, d in enumerate(gop_results):
        status, score, word_model = d
        # if status == False:
        if not status:
            diagnosis_error_index.append(i)

    errindex = 0
    for i, x in enumerate(recommended_sents_res["recommendation"]):
        if len(x["recommend"]) == 0:  # no recommendation at all
            recommended_sentences.append(
                (["n/a"] if i in diagnosis_error_index else [])
            )
        else:
            try:
                sent, lv, ptrn = sorted(x["recommend"])[0]
                cefr_color = CEFR_2_color(lv)
                recommended_sentences.append(
                    [
                        {
                            "en": sent,
                            "level": {"CEFR": lv, "CEFR_color": cefr_color},
                            "grammar": ptrn.split(", "),
                        }
                    ]
                )
            except Exception as e:
                recommended_sentences.append(["n/a"])
        errindex += 1

    for idx in sorted(del_indices + puncts_indices):
        gop_results.insert(idx, ["****"])
        ipa_ans.insert(idx, "****")
        ipa_pred.insert(idx, "****")
        recommended_sentences.insert(idx, ["****"])
        parts_ctm_lists.insert(idx, ["****"])
    if prtmsg:
        print("gop_results", gop_results)
        print("ipa_ans", ipa_ans)
        print("ipa_pred", ipa_pred)
        print("recommended_sentences", recommended_sentences)
    if prtmsg:
        print("======GOP result{E}======")
    return (
        gop_results,
        ipa_ans,
        ipa_pred,
        recommended_sentences,
        parts_ctm_lists,
        total_star,
        gop_score_lvs,
        res,
    )


def do_userwav_ctm(wav_path, detail_parts):
    wav = AudioSegment.from_wav(wav_path)
    # print('wav_path', wav_path)
    wav_bp = "./wavs/ctm"
    wav_path_base = os.path.splitext(wav_path)[0].split(os.sep)[-1]
    parts_ctm_lists = []
    rn = 1
    for dic in detail_parts:
        # print('dic', dic)
        word = dic.get("phone")
        # "intervals":[0.61, 1.05],

        start = dic.get("intervals")[0]
        end = dic.get("intervals")[1]

        wordSeg = wav[float(start) * 1000 : (float(end)) * 1000]
        segFileName = "%s__%s__%s.wav" % (wav_path_base, rn, word)
        wordSeg = wordSeg.set_frame_rate(16000)
        wordSeg = wordSeg.set_channels(1)
        seg_fileabspath = os.path.join("./wavs/ctm", segFileName)
        # print('seg_fileabspath', seg_fileabspath)
        web_path_url = "%s%s/%s" % (
            ENGLISH_WEB_ROOT,
            wav_bp.replace("./", "/"),
            segFileName,
        )
        wordSeg.export(seg_fileabspath, format="wav")
        # print('web_path_url', web_path_url)
        parts_ctm_lists.append(("%s" % word, start, end, "%s" % web_path_url))
        rn += 1

    # print('parts_ctm_lists', parts_ctm_lists)
    return parts_ctm_lists


def generateTTSResult(text):
    """Get TTS api results and convert base64 string of generated audio into file

    Args:
      text: input text for speech synthesis

    Returns:
      wavfile_path: the waveform file path of the generated audio
    """

    error_silent = ApiError(
        code="Unable to generate TTS voice",
        message="Unable to generate TTS voice, please try another sentence.",
    )
    try:
        res = getEngTTS(text)
        # print('res', str(res)[0:1000])
        base64_str = res["base64"]
        converted_text = res["converted_text"]
    except Exception as e:
        raise ApiException(status_code=422, error=error_silent)
    randint = np.random.randint(9999999)
    # wavfile_path = f"wavs/result_{randint}.wav"
    mp3_path = f"mp3/result_{randint}.mp3"

    with open(mp3_path, "wb") as opf:
        decode_string = base64.b64decode(base64_str)
        opf.write(decode_string)
    return mp3_path, converted_text


def check_connected_in_sentence(connected_dict, sentence):
    word_in_sentence = sentence.split()
    seq_len = len(word_in_sentence)

    connected_list = []
    duplicate = []
    for start in range(seq_len - 1):
        for end in range(start + 2, start + 6):
            tmp_connected = " ".join(word_in_sentence[start:end])
            tmp_connected = (
                tmp_connected.replace(",", "").replace(".", "").replace("!", "")
            )
            if not connected_list:
                if connected_dict.get(tmp_connected):
                    connected_list.append([tmp_connected, start, end])
                    # print(end, start)
                    while end - start > 1:
                        duplicate.append(range(start, end))
                        start += 1

            elif (
                connected_dict.get(tmp_connected) and range(start, end) not in duplicate
            ):
                _, last_start, last_end = connected_list[-1]
                if set(range(last_start, last_end)) & set(range(start, end)):
                    connected_list[-1] = [
                        " ".join(word_in_sentence[last_start:end]),
                        last_start,
                        end,
                    ]
                    while end - last_start > 1:
                        duplicate.append(range(last_start, end))
                        last_start += 1
                else:
                    connected_list.append([tmp_connected, start, end])
                    while end - start > 1:
                        duplicate.append(range(start, end))
                        start += 1

    # # start, end pointer
    # start = 0
    # end = 2
    # connected_list = []

    # while end <= seq_len:
    #     tmp_connected = " ".join(word_in_sentence[start:end])
    #     tmp_connected = tmp_connected.replace(",", "").replace(".", "")
    #     if connected_dict.get(tmp_connected):
    #         if connected_list:
    #             if connected_list[-1][0] == tmp_connected:
    #                 connected_list.append([tmp_connected, (start, end)])
    #             elif connected_list[-1][0] in tmp_connected:
    #                 connected_list[-1] = [tmp_connected, (start, end)]
    #             else:
    #                 connected_list.append([tmp_connected, (start, end)])
    #         else:
    #             connected_list.append([tmp_connected, (start, end)])
    #         end += 1
    #     elif not connected_dict.get(tmp_connected):
    #         start += 1
    #         end = start + 2

    return connected_list


def connected_list_post_processing(connected_list, intervals_list, soundfile_data):
    sample_rate = 16000
    # https://ml-dev.ponddy.com/connected_speech/predict
    base_uri = "https://ml-dev.ponddy.com"
    endpoint = "connected_speech/predict"
    acturi = "%s/%s" % (base_uri, endpoint)
    connected_results = []

    if not connected_list:
        connected_results.append(
            {
                "status_code": None,
                "connected_words": None,
                "start_end_word_index": None,
                "label": None,
                "confidence": None,
            }
        )

        return connected_results

    # print("connected_list", connected_list)
    # print(connected_list)
    for connected_metadata in connected_list:
        connected_words, start, end = connected_metadata
        # start, end = start_end_index
        # print("start", start)
        # print("end", end)
        # print("intervals_list", intervals_list)
        end -= 1
        start_sec = intervals_list[start][0]  # take the start sec of the sound
        end_sec = intervals_list[end][-1]  # take the end sec of the sound

        soundfile_data_start_index = math.floor(start_sec * sample_rate)
        soundfile_data_end_index = math.ceil(end_sec * sample_rate)

        part_data = soundfile_data[soundfile_data_start_index:soundfile_data_end_index]

        with tempfile.TemporaryDirectory() as tmp:
            file = f"{tmp}.wav"
            sf.write(file, part_data, sample_rate)

            with open(file, "rb") as f:
                connected_b64 = base64.b64encode(f.read()).decode("UTF-8")

            payloads = {"base64": connected_b64}
            try:
                r = requests.post(
                    acturi, json=payloads, headers={"Connection": "close"}
                )
            except ConnectionError as e:
                _ = e
                return 404, {}

            status_code = r.status_code
            result = json.loads(r.text)
            r.connection.close()
            confidence = result.get("Confidence")
            label = result.get("Label")

            connected_result = {
                "status_code": status_code,
                "connected_words": connected_words,
                "start_end_word_index": [start, end],
                "label": label,
                "confidence": confidence,
            }

            connected_results.append(connected_result)

    print("connected_result", connected_results)
    return connected_results
