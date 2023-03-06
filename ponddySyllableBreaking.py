import os, sys, json
import pickle
import pandas as pd
import traceback
from itertools import chain

## constants
SLAX   = {'IH1', 'IH2', 'EH1', 'EH2', 'AE1', 'AE2', 'AH1', 'AH2',
                                                    'UH1', 'UH2',}
VOWELS = {'IY1', 'IY2', 'IY0', 'EY1', 'EY2', 'EY0', 'AA1', 'AA2', 'AA0',
          'ER1', 'ER2', 'ER0', 'AW1', 'AW2', 'AW0', 'AO1', 'AO2', 'AO0',
          'AY1', 'AY2', 'AY0', 'OW1', 'OW2', 'OW0', 'OY1', 'OY2', 'OY0',
          'IH0', 'EH0', 'AE0', 'AH0', 'UH0', 'UW1', 'UW2', 'UW0', 'UW',
          'IY',  'EY',  'AA',  'ER',   'AW', 'AO',  'AY',  'OW',  'OY',
          'UH',  'IH',  'EH',  'AE',  'AH',  'UH',} | SLAX

## licit medial onsets

O2 = {('P', 'R'), ('T', 'R'), ('K', 'R'), ('B', 'R'), ('D', 'R'),
      ('G', 'R'), ('F', 'R'), ('TH', 'R'),
      ('P', 'L'), ('K', 'L'), ('B', 'L'), ('G', 'L'),
      ('F', 'L'), ('S', 'L'),
      ('K', 'W'), ('G', 'W'), ('S', 'W'),
      ('S', 'P'), ('S', 'T'), ('S', 'K'),
      ('HH', 'Y'), # "clerihew"
      ('R', 'W'),}
O3 = {('S', 'T', 'R'), ('S', 'K', 'L'), ('T', 'R', 'W')} # "octroi"

# This does not represent anything like a complete list of onsets, but
# merely those that need to be maximized in medial position.

def syllabify(pron, alaska_rule=True):
    """
    Syllabifies a CMU dictionary (ARPABET) word string

    # Alaska rule:
    >>> pprint(syllabify('AH0 L AE1 S K AH0'.split())) # Alaska
    '-AH0-.L-AE1-S.K-AH0-'
    >>> pprint(syllabify('AH0 L AE1 S K AH0'.split(), 0)) # Alaska
    '-AH0-.L-AE1-.S K-AH0-'

    # huge medial onsets:
    >>> pprint(syllabify('M IH1 N S T R AH0 L'.split())) # minstrel
    'M-IH1-N.S T R-AH0-L'
    >>> pprint(syllabify('AA1  K T R W AA0 R'.split())) # octroi
    '-AA1-K.T R W-AA0-R'

    # destressing
    >>> pprint(destress(syllabify('M IH1 L AH0 T EH2 R IY0'.split())))
    'M-IH-.L-AH-.T-EH-.R-IY-'

    # normal treatment of 'j':
    >>> pprint(syllabify('M EH1 N Y UW0'.split())) # menu
    'M-EH1-N.Y-UW0-'
    >>> pprint(syllabify('S P AE1 N Y AH0 L'.split())) # spaniel
    'S P-AE1-N.Y-AH0-L'
    >>> pprint(syllabify('K AE1 N Y AH0 N'.split())) # canyon
    'K-AE1-N.Y-AH0-N'
    >>> pprint(syllabify('M IH0 N Y UW2 EH1 T'.split())) # minuet
    'M-IH0-N.Y-UW2-.-EH1-T'
    >>> pprint(syllabify('JH UW1 N Y ER0'.split())) # junior
    'JH-UW1-N.Y-ER0-'
    >>> pprint(syllabify('K L EH R IH HH Y UW'.split())) # clerihew
    'K L-EH-.R-IH-.HH Y-UW-'

    # nuclear treatment of 'j'
    >>> pprint(syllabify('R EH1 S K Y UW0'.split())) # rescue
    'R-EH1-S.K-Y UW0-'
    >>> pprint(syllabify('T R IH1 B Y UW0 T'.split())) # tribute
    'T R-IH1-B.Y-UW0-T'
    >>> pprint(syllabify('N EH1 B Y AH0 L AH0'.split())) # nebula
    'N-EH1-B.Y-AH0-.L-AH0-'
    >>> pprint(syllabify('S P AE1 CH UH0 L AH0'.split())) # spatula
    'S P-AE1-.CH-UH0-.L-AH0-'
    >>> pprint(syllabify('AH0 K Y UW1 M AH0 N'.split())) # acumen
    '-AH0-K.Y-UW1-.M-AH0-N'
    >>> pprint(syllabify('S AH1 K Y AH0 L IH0 N T'.split())) # succulent
    'S-AH1-K.Y-AH0-.L-IH0-N T'
    >>> pprint(syllabify('F AO1 R M Y AH0 L AH0'.split())) # formula
    'F-AO1 R-M.Y-AH0-.L-AH0-'
    >>> pprint(syllabify('V AE1 L Y UW0'.split())) # value
    'V-AE1-L.Y-UW0-'

    # everything else
    >>> pprint(syllabify('N AO0 S T AE1 L JH IH0 K'.split())) # nostalgic
    'N-AO0-.S T-AE1-L.JH-IH0-K'
    >>> pprint(syllabify('CH ER1 CH M AH0 N'.split())) # churchmen
    'CH-ER1-CH.M-AH0-N'
    >>> pprint(syllabify('K AA1 M P AH0 N S EY2 T'.split())) # compensate
    'K-AA1-M.P-AH0-N.S-EY2-T'
    >>> pprint(syllabify('IH0 N S EH1 N S'.split())) # inCENSE
    '-IH0-N.S-EH1-N S'
    >>> pprint(syllabify('IH1 N S EH2 N S'.split())) # INcense
    '-IH1-N.S-EH2-N S'
    >>> pprint(syllabify('AH0 S EH1 N D'.split())) # ascend
    '-AH0-.S-EH1-N D'
    >>> pprint(syllabify('R OW1 T EY2 T'.split())) # rotate
    'R-OW1-.T-EY2-T'
    >>> pprint(syllabify('AA1 R T AH0 S T'.split())) # artist
    '-AA1 R-.T-AH0-S T'
    >>> pprint(syllabify('AE1 K T ER0'.split())) # actor
    '-AE1-K.T-ER0-'
    >>> pprint(syllabify('P L AE1 S T ER0'.split())) # plaster
    'P L-AE1-S.T-ER0-'
    >>> pprint(syllabify('B AH1 T ER0'.split())) # butter
    'B-AH1-.T-ER0-'
    >>> pprint(syllabify('K AE1 M AH0 L'.split())) # camel
    'K-AE1-.M-AH0-L'
    >>> pprint(syllabify('AH1 P ER0'.split())) # upper
    '-AH1-.P-ER0-'
    >>> pprint(syllabify('B AH0 L UW1 N'.split())) # balloon
    'B-AH0-.L-UW1-N'
    >>> pprint(syllabify('P R OW0 K L EY1 M'.split())) # proclaim
    'P R-OW0-.K L-EY1-M'
    >>> pprint(syllabify('IH0 N S EY1 N'.split())) # insane
    '-IH0-N.S-EY1-N'
    >>> pprint(syllabify('IH0 K S K L UW1 D'.split())) # exclude
    '-IH0-K.S K L-UW1-D'
    """
    ## main pass
    mypron = list(pron)
    nuclei = []
    onsets = []
    i = -1
    for (j, seg) in enumerate(mypron):
        if seg in VOWELS:
            nuclei.append([seg])
            onsets.append(mypron[i + 1:j]) # actually interludes, r.n.
            i = j
    codas = [mypron[i + 1:]]
    ## resolve disputes and compute coda
    for i in range(1, len(onsets)):
        coda = []
        # boundary cases
        if len(onsets[i]) > 1 and onsets[i][0] == 'R':
            nuclei[i - 1].append(onsets[i].pop(0))
        if len(onsets[i]) > 2 and onsets[i][-1] == 'Y':
            nuclei[i].insert(0, onsets[i].pop())
        if len(onsets[i]) > 1 and alaska_rule and nuclei[i-1][-1] in SLAX \
                                              and onsets[i][0] == 'S':
            coda.append(onsets[i].pop(0))
        # onset maximization
        depth = 1
        if len(onsets[i]) > 1:
            if tuple(onsets[i][-2:]) in O2:
                depth = 3 if tuple(onsets[i][-3:]) in O3 else 2
        for j in range(len(onsets[i]) - depth):
            coda.append(onsets[i].pop(0))
        # store coda
        codas.insert(i - 1, coda)

    ## verify that all segments are included in the ouput
    output = list(zip(onsets, nuclei, codas))  # in Python3 zip is a generator
    flat_output = list(chain.from_iterable(chain.from_iterable(output)))
    if flat_output != mypron:
        raise ValueError(f"could not syllabify {mypron}, got {flat_output}")
    return output


def pprint(syllab):
    """
    Pretty-print a syllabification
    """
    return '.'.join('-'.join(' '.join(p) for p in syl) for syl in syllab)


def destress(syllab):
    """
    Generate a syllabification with nuclear stress information removed
    """
    syls = []
    for (onset, nucleus, coda) in syllab:
        nuke = [p[:-1] if p[-1] in {'0', '1', '2'} else p for p in nucleus]
        syls.append((onset, nuke, coda))
    return syls

# with open('ponddyWordSyllableDict.pkl', 'rb') as f:
#     ponddyWordSyllableTable = pickle.load(f)

phoneToIPA = {'AO': 'ɔː',
             'AO0': 'ɔː',
             'AO1': 'ɔː',
             'AO2': 'ɔː',
             'AA': 'ɑː',
             'AA0': 'ɑː',
             'AA1': 'ɑː',
             'AA2': 'ɑː',
             'IY': 'i',
             'IY0': 'i',
             'IY1': 'iː',
             'IY2': 'iː',
             'UW': 'uː',
             'UW0': 'uː',
             'UW1': 'uː',
             'UW2': 'uː',
             'EH': 'e',
             'EH0': 'e',
             'EH1': 'e',
             'EH2': 'e',
             'IH': 'ɪ',
             'IH0': 'ɪ',
             'IH1': 'ɪ',
             'IH2': 'ɪ',
             'UH': 'ʊ',
             'UH0': 'ʊ',
             'UH1': 'ʊ',
             'UH2': 'ʊ',
             'AH': 'ʌ',
             'AH0': 'ə',
             'AH1': 'ʌ',
             'AH2': 'ʌ',
             'AE': 'æ',
             'AE0': 'æ',
             'AE1': 'æ',
             'AE2': 'æ',
             'AX': 'ə',
             'AX0': 'ə',
             'AX1': 'ə',
             'AX2': 'ə',
             'EY': 'eɪ',
             'EY0': 'eɪ',
             'EY1': 'eɪ',
             'EY2': 'eɪ',
             'AY': 'aɪ',
             'AY0': 'aɪ',
             'AY1': 'aɪ',
             'AY2': 'aɪ',
             'OW': 'oʊ',
             'OW0': 'oʊ',
             'OW1': 'oʊ',
             'OW2': 'oʊ',
             'AW': 'aʊ',
             'AW0': 'aʊ',
             'AW1': 'aʊ',
             'AW2': 'aʊ',
             'OY': 'ɔɪ',
             'OY0': 'ɔɪ',
             'OY1': 'ɔɪ',
             'OY2': 'ɔɪ',
             'P': 'p',
             'B': 'b',
             'T': 't',
             'D': 'd',
             'K': 'k',
             'G': 'g',
             'CH': 'ʧ',
             'JH': 'ʤ',
             'F': 'f',
             'V': 'v',
             'TH': 'θ',
             'DH': 'ð',
             'S': 's',
             'Z': 'z',
             'SH': 'ʃ',
             'ZH': 'ʒ',
             'HH': 'h',
             'M': 'm',
             'N': 'n',
             'NG': 'ŋ',
             'L': 'l',
             'R': 'r',
             'ER': 'ɜːr',
             'ER0': 'ər',
             'ER1': 'ɜːr',
             'ER2': 'ɜːr',
             'AXR': 'ər',
             'AXR0': 'ər',
             'AXR1': 'ər',
             'AXR2': 'ər',
             'W': 'w',
             'Y': 'j'}

with open('ponddyWordSyllableDict.pkl', 'rb') as f:
    ponddyWordSyllableTable = pickle.load(f)
    
def getSyllableBreakingSingleUtt(uttGOP_in):
    def getStressedBestCMU(ans_ph, pred_ph):
        tmp = []
        for x, y in zip(ans_ph, pred_ph):
            if x in y:
                tmp.append(x)
            elif any(v in x for v in 'AEIOU'):
                skip = False
                for yy in y:
                    if yy[:2] == x[:2]:
                        tmp.append(yy)
                        skip = True
                        break
                if not skip:
                    tmp.append(y[0])
            else:
                tmp.append(y[0])
        return tmp
    def getSyllableStress(cmu_phs):
        for ph in cmu_phs:
            if ph != 'SIL' and any(v in ph for v in 'AEIOU'):
                if len(ph) == 3:
                    return int(ph[2])
                else:
                    return 0
        return 0
    
    def simplifyStress(stress_in):
        ret = None
        stress_set = set(stress_in)
        if 1 in stress_set:
            ret = [1 if x == 1 else 0 for x in stress_in]
        elif 2 in stress_set and 0 in stress_set:
            ret = [1 if x == 2 else 0 for x in stress_in]
        else:
            ret = [0 for x in stress_in]
        return ret
    
    uttGOP = uttGOP_in.copy()
    words = []
    for w in uttGOP['parts']:
        cur = {}
        cur['word'] = w['word']
        phone = tuple(w['phone'].split('_'))
        phone_scores = w['rawRRScores']
        
        syllable_info = ponddyWordSyllableTable.get(w['word'].lower(), {}).get(phone, {})
        
        try:
            syllable_phones = syllabify(phone)
        except:
            syllable_phones = [phone]
        syllable_phones = [[x for ssub in sub for x in ssub if len(ssub) > 0] for sub in syllable_phones]
#         print('syllable_phones:', syllable_phones)
        
        predPhoneStressedCMU = getStressedBestCMU(w['phone'].split('_'), w['predPhone'])
        pred_phones = [phoneToIPA.get(x, '*') for x in w['predPhoneBestCMU']]
        ans_phones = [phoneToIPA.get(x, '*') for x in w['ansCMU']]
        ps, ss = "ˈ", "ˌ"
        try:
            syllable_ipa_pairs = syllable_info.get('syllable_ipa_pairs', [])
            syallblify_phone_scores = []
            syllablify_pred_phones = []
            syllablify_ans_phones = []
            k = 0
            ans_stress, pred_stress = [], []
            for y in syllable_phones:
                syallblify_phone_scores.append(phone_scores[k:k+len(y)])
                syllablify_pred_phones.append(pred_phones[k:k+len(y)])
                syllablify_ans_phones.append(ans_phones[k:k+len(y)])
                ans_st = getSyllableStress(w['ansCMU'][k:k+len(y)])
                pred_st = getSyllableStress(predPhoneStressedCMU[k:k+len(y)])
                ans_stress.append(ans_st)
                pred_stress.append(pred_st)
                k += len(y)
            
            ans_stress = simplifyStress(ans_stress)
            pred_stress = simplifyStress(pred_stress)
            
            stress = [0] if len(ans_stress) == 1 else ans_stress
            ipa_break_merged = [(ps if y == 1 else (ss if y == 2 else ""))+''.join(x) for x, y in zip(syllablify_ans_phones, stress)]
            ipa_break = syllablify_ans_phones
            stress = [0] if len(pred_stress) == 1 else pred_stress
            ipa_break_pred_merged = [(ps if y == 1 else (ss if y == 2 else ""))+''.join(x) for x, y in zip(syllablify_pred_phones, stress)]
            
            if len(syllable_ipa_pairs) > 0:
                syllable = [x[0] for x in syllable_ipa_pairs]
            
            cur['syllable'] = syllable if ''.join(syllable) == cur['word'].lower() else [cur['word'].lower()]
            cur['ipa_break_merged'] = ipa_break_merged
            cur['ipa_break'] = ipa_break
            cur['ipa_break_pred'] = syllablify_pred_phones
            cur['ipa_break_pred_merged'] = ipa_break_pred_merged
            cur['phone_scores'] = syallblify_phone_scores    
            cur['ans_stress'] = ans_stress if len(ans_stress) > 1 else [1]
            cur['pred_stress'] = pred_stress if len(pred_stress) > 1 else [1]
        except Exception as e:
            cur['syllable'] = [w['word'].lower()]
            cur['ipa_break_merged'] = [w['ipaAns']]
            cur['ipa_break'] = [[phoneToIPA.get(x, '*') for x in w['ansCMU']]]
            cur['ipa_break_pred'] = [[phoneToIPA.get(x, '*') for x in w['predPhoneBestCMU']]]
            cur['ipa_break_pred_merged'] = [[phoneToIPA.get(x, '*') for x in w['predPhoneBestCMU']]]
            cur['phone_scores'] = [phone_scores]
            cur['ans_stress'] = [0]
            cur['pred_stress'] = [0]
        words.append(cur)
    return words

def getSyllableBreakingAllUtts(uttGOPs):
    ret = {}
    for uttid in uttGOPs:
        words = getSyllableBreakingSingleUtt(uttGOPs[uttid])
        ret[uttid] = words
    return ret
    
if __name__ == '__main__':
    test_uttGOP = {'RR_scores': 90,
 'RR_total': 11,
 'RR_correct': 9,
 'GOP_scores': 86,
 'GOP_total': 11,
 'GOP_correct': 7,
 'parts': [{'word': 'THE',
   'phone': 'DH_AH0',
   'rawGOP': (-0.135496, -0.3239594),
   'rawRR': (0.25, 0.03333333333333333),
   'rawGOPScores': [7.3602736621956, 3.0853369942536144],
   'rawRRScores': [3.9968038348871575, 28.73478855663454],
   'GOPScore': 5.222805328224608,
   'RRScore': 16.36579619576085,
   'GOPScore_min': 3.0853369942536144,
   'RRScore_min': 3.9968038348871575,
   'phoneIntervals': ((0.0, 0.78), (0.78, 0.86)),
   'predPhone': (('SIL', 'D', 'P'), ('EH2', 'EH1', 'EH0')),
   'intervals': (0.0, 0.86),
   'RR_Result': 'false',
   'GOP_Result': 'false',
   'ipaAns': 'ðʌ',
   'ipaAnsNoStress': 'ðʌ',
   'ansIpaDiffIndex': [0, 1],
   'ansCMU': ['DH', 'AH0'],
   'ansCMUDiffIndex': [0, 1],
   'predPhoneBestIPA': '*e',
   'predIpaDiffIndex': [0, 1],
   'predPhoneBestCMU': ['SIL', 'EH2'],
   'predPhoneDiffIndex': [0, 1],
   'predPhoneError': (2, 2),
   'wordScore': 0.0},
  {'word': 'NEIGHBOR',
   'phone': 'N_EY1_B_ER0',
   'rawGOP': (0.0, 0.0, -0.3113077, 0.0),
   'rawRR': (0.0, 0.0, 0.0, 0.0),
   'rawGOPScores': [100.0, 100.0, 3.2105998969163396, 100.0],
   'rawRRScores': [100.0, 100.0, 100.0, 100.0],
   'GOPScore': 75.80264997422908,
   'RRScore': 100.0,
   'GOPScore_min': 3.2105998969163396,
   'RRScore_min': 100.0,
   'phoneIntervals': ((0.86, 1.02), (1.02, 1.2), (1.2, 1.27), (1.27, 1.42)),
   'predPhone': (('N', 'SPN', 'D'),
    ('EY2', 'EY1', 'EY0'),
    ('P', 'B', 'SPN'),
    ('ER0', 'ER2', 'ER1')),
   'intervals': (0.86, 1.42),
   'RR_Result': 'true',
   'GOP_Result': 'false',
   'ipaAns': 'neɪbɜːr',
   'ipaAnsNoStress': 'neɪbɜːr',
   'ansIpaDiffIndex': [],
   'ansCMU': ['N', 'EY1', 'B', 'ER0'],
   'ansCMUDiffIndex': [],
   'predPhoneBestIPA': 'neɪbɜːr',
   'predIpaDiffIndex': [],
   'predPhoneBestCMU': ['N', 'EY1', 'B', 'ER0'],
   'predPhoneDiffIndex': [],
   'predPhoneError': (0, 4),
   'wordScore': 100.0},
  {'word': 'COMPLAINED',
   'phone': 'K_AH0_M_P_L_EY1_N_D',
   'rawGOP': (0.0, -0.1198085, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
   'rawRR': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
   'rawGOPScores': [100.0,
    8.317730080805731,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0],
   'rawRRScores': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
   'GOPScore': 88.53971626010072,
   'RRScore': 100.0,
   'GOPScore_min': 8.317730080805731,
   'RRScore_min': 100.0,
   'phoneIntervals': ((1.42, 1.5),
    (1.5, 1.61),
    (1.61, 1.69),
    (1.69, 1.79),
    (1.79, 1.87),
    (1.87, 2.04),
    (2.04, 2.11),
    (2.11, 2.14)),
   'predPhone': (('K', 'SPN', 'SIL'),
    ('AH2', 'AH1', 'AH'),
    ('M', 'SPN', 'N'),
    ('P', 'HH', 'B'),
    ('L', 'SPN', 'W'),
    ('EY2', 'EY1', 'EY0'),
    ('N', 'M', 'NG'),
    ('D', 'N', 'T')),
   'intervals': (1.42, 2.14),
   'RR_Result': 'true',
   'GOP_Result': 'false',
   'ipaAns': 'kʌmpleɪnd',
   'ipaAnsNoStress': 'kʌmpleɪnd',
   'ansIpaDiffIndex': [],
   'ansCMU': ['K', 'AH0', 'M', 'P', 'L', 'EY1', 'N', 'D'],
   'ansCMUDiffIndex': [],
   'predPhoneBestIPA': 'kʌmpleɪnd',
   'predIpaDiffIndex': [],
   'predPhoneBestCMU': ['K', 'AH0', 'M', 'P', 'L', 'EY1', 'N', 'D'],
   'predPhoneDiffIndex': [],
   'predPhoneError': (0, 8),
   'wordScore': 100.0},
  {'word': 'TO',
   'phone': 'T_IH0',
   'rawGOP': (0.0, 0.0),
   'rawRR': (0.0, 0.0),
   'rawGOPScores': [100.0, 100.0],
   'rawRRScores': [100.0, 100.0],
   'GOPScore': 100.0,
   'RRScore': 100.0,
   'GOPScore_min': 100.0,
   'RRScore_min': 100.0,
   'phoneIntervals': ((2.14, 2.22), (2.22, 2.28)),
   'predPhone': (('T', 'HH', 'CH'), ('IH0', 'IH2', 'IH1')),
   'intervals': (2.14, 2.28),
   'RR_Result': 'true',
   'GOP_Result': 'true',
   'ipaAns': 'tɪ',
   'ipaAnsNoStress': 'tɪ',
   'ansIpaDiffIndex': [],
   'ansCMU': ['T', 'IH0'],
   'ansCMUDiffIndex': [],
   'predPhoneBestIPA': 'tɪ',
   'predIpaDiffIndex': [],
   'predPhoneBestCMU': ['T', 'IH0'],
   'predPhoneDiffIndex': [],
   'predPhoneError': (0, 2),
   'wordScore': 100.0},
  {'word': 'ME',
   'phone': 'M_IY1',
   'rawGOP': (0.0, 0.0),
   'rawRR': (0.0, 0.0),
   'rawGOPScores': [100.0, 100.0],
   'rawRRScores': [100.0, 100.0],
   'GOPScore': 100.0,
   'RRScore': 100.0,
   'GOPScore_min': 100.0,
   'RRScore_min': 100.0,
   'phoneIntervals': ((2.28, 2.38), (2.38, 2.64)),
   'predPhone': (('M', 'SPN', 'N'), ('IY2', 'IY1', 'IY0')),
   'intervals': (2.28, 2.64),
   'RR_Result': 'true',
   'GOP_Result': 'true',
   'ipaAns': 'miː',
   'ipaAnsNoStress': 'miː',
   'ansIpaDiffIndex': [],
   'ansCMU': ['M', 'IY1'],
   'ansCMUDiffIndex': [],
   'predPhoneBestIPA': 'miː',
   'predIpaDiffIndex': [],
   'predPhoneBestCMU': ['M', 'IY1'],
   'predPhoneDiffIndex': [],
   'predPhoneError': (0, 2),
   'wordScore': 100.0},
  {'word': 'ABOUT',
   'phone': 'AH0_B_AW1_T',
   'rawGOP': (0.0, 0.0, 0.0, 0.0),
   'rawRR': (0.0, 0.0, 0.0, 0.0),
   'rawGOPScores': [100.0, 100.0, 100.0, 100.0],
   'rawRRScores': [100.0, 100.0, 100.0, 100.0],
   'GOPScore': 100.0,
   'RRScore': 100.0,
   'GOPScore_min': 100.0,
   'RRScore_min': 100.0,
   'phoneIntervals': ((2.7, 2.8), (2.8, 2.88), (2.88, 3.01), (3.01, 3.08)),
   'predPhone': (('AH0', 'AH2', 'AH1'),
    ('B', 'SPN', 'V'),
    ('AW2', 'AW1', 'AW0'),
    ('T', 'P', 'SPN')),
   'intervals': (2.7, 3.08),
   'RR_Result': 'true',
   'GOP_Result': 'true',
   'ipaAns': 'ʌbaʊt',
   'ipaAnsNoStress': 'ʌbaʊt',
   'ansIpaDiffIndex': [],
   'ansCMU': ['AH0', 'B', 'AW1', 'T'],
   'ansCMUDiffIndex': [],
   'predPhoneBestIPA': 'ʌbaʊt',
   'predIpaDiffIndex': [],
   'predPhoneBestCMU': ['AH0', 'B', 'AW1', 'T'],
   'predPhoneDiffIndex': [],
   'predPhoneError': (0, 4),
   'wordScore': 100.0},
  {'word': 'THE',
   'phone': 'DH_AH0',
   'rawGOP': (0.0, 0.0),
   'rawRR': (0.0, 0.0),
   'rawGOPScores': [100.0, 100.0],
   'rawRRScores': [100.0, 100.0],
   'GOPScore': 100.0,
   'RRScore': 100.0,
   'GOPScore_min': 100.0,
   'RRScore_min': 100.0,
   'phoneIntervals': ((3.08, 3.11), (3.11, 3.16)),
   'predPhone': (('DH', 'B', 'V'), ('AH0', 'IH2', 'IH1')),
   'intervals': (3.08, 3.16),
   'RR_Result': 'true',
   'GOP_Result': 'true',
   'ipaAns': 'ðʌ',
   'ipaAnsNoStress': 'ðʌ',
   'ansIpaDiffIndex': [],
   'ansCMU': ['DH', 'AH0'],
   'ansCMUDiffIndex': [],
   'predPhoneBestIPA': 'ðʌ',
   'predIpaDiffIndex': [],
   'predPhoneBestCMU': ['DH', 'AH0'],
   'predPhoneDiffIndex': [],
   'predPhoneError': (0, 2),
   'wordScore': 100.0},
  {'word': 'NOISE',
   'phone': 'N_OY1_Z',
   'rawGOP': (0.0, -0.0006789157, 0.0),
   'rawRR': (0.0, 0.0, 0.0),
   'rawGOPScores': [100.0, 99.7703303880105, 100.0],
   'rawRRScores': [100.0, 100.0, 100.0],
   'GOPScore': 99.92344346267016,
   'RRScore': 100.0,
   'GOPScore_min': 99.7703303880105,
   'RRScore_min': 100.0,
   'phoneIntervals': ((3.16, 3.33), (3.33, 3.71), (3.71, 3.83)),
   'predPhone': (('N', 'SPN', 'D'), ('OY2', 'OY0', 'OY'), ('Z', 'S', 'SPN')),
   'intervals': (3.16, 3.83),
   'RR_Result': 'true',
   'GOP_Result': 'true',
   'ipaAns': 'nɔɪz',
   'ipaAnsNoStress': 'nɔɪz',
   'ansIpaDiffIndex': [],
   'ansCMU': ['N', 'OY1', 'Z'],
   'ansCMUDiffIndex': [],
   'predPhoneBestIPA': 'nɔɪz',
   'predIpaDiffIndex': [],
   'predPhoneBestCMU': ['N', 'OY1', 'Z'],
   'predPhoneDiffIndex': [],
   'predPhoneError': (0, 3),
   'wordScore': 100.0},
  {'word': 'FROM',
   'phone': 'F_R_AH1_M',
   'rawGOP': (0.0, 0.0, -0.02396879, 0.0),
   'rawRR': (0.0, 0.0, 0.03333333333333333, 0.0),
   'rawGOPScores': [100.0, 100.0, 38.50419882830409, 100.0],
   'rawRRScores': [100.0, 100.0, 28.73478855663454, 100.0],
   'GOPScore': 84.62604970707602,
   'RRScore': 82.18369713915864,
   'GOPScore_min': 38.50419882830409,
   'RRScore_min': 28.73478855663454,
   'phoneIntervals': ((3.83, 4.0), (4.0, 4.08), (4.08, 4.18), (4.18, 4.34)),
   'predPhone': (('F', 'TH', 'SPN'),
    ('R', 'SPN', 'S'),
    ('AA2', 'AA1', 'AA0'),
    ('M', 'SPN', 'N')),
   'intervals': (3.83, 4.34),
   'RR_Result': 'false',
   'GOP_Result': 'false',
   'ipaAns': 'frʌm',
   'ipaAnsNoStress': 'frʌm',
   'ansIpaDiffIndex': [2],
   'ansCMU': ['F', 'R', 'AH1', 'M'],
   'ansCMUDiffIndex': [2],
   'predPhoneBestIPA': 'frɑːm',
   'predIpaDiffIndex': [2, 3],
   'predPhoneBestCMU': ['F', 'R', 'AA2', 'M'],
   'predPhoneDiffIndex': [2],
   'predPhoneError': (1, 4),
   'wordScore': 75.0},
  {'word': 'THE',
   'phone': 'DH_AH0',
   'rawGOP': (0.0, 0.0),
   'rawRR': (0.0, 0.0),
   'rawGOPScores': [100.0, 100.0],
   'rawRRScores': [100.0, 100.0],
   'GOPScore': 100.0,
   'RRScore': 100.0,
   'GOPScore_min': 100.0,
   'RRScore_min': 100.0,
   'phoneIntervals': ((4.34, 4.38), (4.38, 4.46)),
   'predPhone': (('DH', 'V', 'F'), ('AH0', 'AH2', 'AH1')),
   'intervals': (4.34, 4.46),
   'RR_Result': 'true',
   'GOP_Result': 'true',
   'ipaAns': 'ðʌ',
   'ipaAnsNoStress': 'ðʌ',
   'ansIpaDiffIndex': [],
   'ansCMU': ['DH', 'AH0'],
   'ansCMUDiffIndex': [],
   'predPhoneBestIPA': 'ðʌ',
   'predIpaDiffIndex': [],
   'predPhoneBestCMU': ['DH', 'AH0'],
   'predPhoneDiffIndex': [],
   'predPhoneError': (0, 2),
   'wordScore': 100.0},
  {'word': 'DOG',
   'phone': 'D_AO1_G',
   'rawGOP': (0.0, 0.0, 0.0),
   'rawRR': (0.0, 0.0, 0.0),
   'rawGOPScores': [100.0, 100.0, 100.0],
   'rawRRScores': [100.0, 100.0, 100.0],
   'GOPScore': 100.0,
   'RRScore': 100.0,
   'GOPScore_min': 100.0,
   'RRScore_min': 100.0,
   'phoneIntervals': ((4.46, 4.58), (4.58, 4.95), (4.95, 5.08)),
   'predPhone': (('D', 'SPN', 'JH'), ('AO1', 'AO2', 'AO0'), ('G', 'K', 'B')),
   'intervals': (4.46, 5.08),
   'RR_Result': 'true',
   'GOP_Result': 'true',
   'ipaAns': 'dɔːg',
   'ipaAnsNoStress': 'dɔːg',
   'ansIpaDiffIndex': [],
   'ansCMU': ['D', 'AO1', 'G'],
   'ansCMUDiffIndex': [],
   'predPhoneBestIPA': 'dɔːg',
   'predIpaDiffIndex': [],
   'predPhoneBestCMU': ['D', 'AO1', 'G'],
   'predPhoneDiffIndex': [],
   'predPhoneError': (0, 3),
   'wordScore': 100.0}],
 'sentScores': 88.63636363636364}
    ret = getSyllableBreakingSingleUtt(test_uttGOP)
    print(ret)