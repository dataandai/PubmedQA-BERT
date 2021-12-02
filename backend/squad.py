from transformers import BertTokenizer
import torch

from transformers import BertForQuestionAnswering

# a QA feladatoknál a BÍERT-large jobban teljesít, mint a base modellek
model = BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')
# GPU-val, vagy anélkül
cuda = False
if torch.cuda.is_available():
    model.cuda()
    cuda = True


tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')

#  a kérdés szövege és a max 50 absztrakt szövege kerül a BERT modellbe
def get_answer(q, texts):

    answer = ""
    score = 0
    for text in texts:
        i = 0
        tLen = len(text)
        while i + 1024 < tLen:
            perm_answer, perm_score = do_squad(q, text[i:i+1024])
            if perm_score > score:
                score = perm_score
                answer = perm_answer
            i += 1000

    return answer


def do_squad(q, doc):

    input_ids = tokenizer.encode(q, doc)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    assert len(segment_ids) == len(input_ids)

    if cuda:
        start_scores, end_scores = model(torch.tensor([input_ids]).to("cuda"),
                                         token_type_ids=torch.tensor([segment_ids]).to("cuda"))
    else:
        start_scores, end_scores = model(torch.tensor([input_ids]),
                                         token_type_ids=torch.tensor([segment_ids]))

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = ' '.join(tokens[answer_start:answer_end+1])
    first_token = answer_start.data.item()
    last_token = answer_end.data.item()

    if start_scores[0][answer_start] > 0 and end_scores[0][answer_end] > 0 and last_token > first_token:

        start_scores_q = torch.tensor(start_scores).narrow(
            1, first_token, last_token + 1 - first_token)
        end_scores_q = torch.tensor(end_scores).narrow(
            1, first_token, last_token + 1 - first_token)

        if(first_token == last_token):
            score = start_scores_q[0][0].data.item()
        else:
            score = (start_scores_q[0][0].data.item(
            ) + end_scores_q[0][last_token-first_token].data.item())/2
    else:
        return "", 0

    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):
# összefűzi a token darabokat, amelyek ## kezdődnek
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]

    return answer, score
