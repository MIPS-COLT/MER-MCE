# cause path
submit_cause_file_path = "results/submit_all_cause_ck6_wd5_now-n_w-e.txt"
submit_cause_emo_list = [x.strip().split('\t') for x in open(submit_cause_file_path)]
submit_cause_emo_dict = {item[0]: item[1] for item in submit_cause_emo_list}

submit_pred_file_path = "results/submit_now-n_w-e.txt"
submit_pred_emo_list = [x.strip().split(' ') for x in open(submit_pred_file_path)]
submit_pred_emo_dict = {item[0]: item[1] for item in submit_pred_emo_list}

import difflib
def calculate_similarity(str1, str2):
    # 计算字符串相似度，这里使用了 difflib.SequenceMatcher 模块
    # 可根据实际需求选择其他相似度计算方法
    similarity_ratio = difflib.SequenceMatcher(None, str1, str2).ratio()
    return similarity_ratio

def find_most_similar_answer(answer_list, test_str):
    max_similarity = 0
    most_similar_answer = None

    max_similarity_idx = -1
    for i in range(len(answer_list)):
        similarity = calculate_similarity(answer_list[i], test_str)
        # print(similarity)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_answer = answer_list[i]
            max_similarity_idx = i

    return most_similar_answer, max_similarity_idx + 1


import json
submit_json_file_path = "Subtask_2_test.json" 

with open(submit_json_file_path, "r") as file:
    submit_data = json.load(file)

idx_emotion = dict(zip(range(7), ['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']))

# 遍历json：如果有情绪，就读出cause，制作pair
for conv in submit_data:
    utt_submit_list = []
    conv_pred_pairs_list = []
    for utt in conv['conversation']:
        utterance_name = utt['video_name'].split('.')[0]
        utterance_ID = int(utt['utterance_ID'])
        utt_submit_list.append(utt['text'])
        
        pred_emotion_idx = int(submit_pred_emo_dict[utterance_name])
        pred_emotion = idx_emotion[pred_emotion_idx]
        if pred_emotion_idx == 0:
            continue
        
        pred_cause_str =  submit_cause_emo_dict[utterance_name]
        pred_cause, pred_cause_idx = find_most_similar_answer(utt_submit_list, pred_cause_str) 
        if pred_cause_idx == 0:
            pred_cause_idx = utterance_ID
        if utterance_ID != pred_cause_idx:
            # print(utterance_ID, pred_emotion, pred_cause_idx)
            utt_emo = str(utterance_ID) + '_' + pred_emotion
            utt_cause = str(utterance_ID) # pred_cause_idx
            pair = [utt_emo, utt_cause]
            conv_pred_pairs_list.append(pair)            
            print(utterance_ID, pred_emotion, pred_cause_idx, pair)
            
        utt_emo = str(utterance_ID) + '_' + pred_emotion
        utt_cause = str(pred_cause_idx) # pred_cause_idx
        pair = [utt_emo, utt_cause]
        conv_pred_pairs_list.append(pair)
    conv['emotion-cause_pairs'] = conv_pred_pairs_list

submit_json_str = json.dumps(submit_data, indent=4)

save_name = "results/submit_all_cause_ck6_wd5_now-n_w-e.json"
with open(save_name, "w") as file:
    file.write(submit_json_str)
