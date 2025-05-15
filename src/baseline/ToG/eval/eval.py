import argparse
from utils import *
from datasets import load_dataset

def prepare_dataset(dataset_name: str):
    dataset = load_dataset(dataset_name, split="test")
    data = [dict(item) for item in dataset]
    return data, "question"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--output_file", type=str,
                        default="ToG_QoG.json", help="the output file name.")
    parser.add_argument("--constraints_refuse", type=bool,
                        default=True, help="LLM may have refuse erorr, enable this option to skip current sample.")
    args = parser.parse_args()

    datas, question_string = prepare_dataset("lianglz/KGQAGen-10k")
    print("test!")

    with open(args.output_file, encoding='utf-8') as f:
        output_datas = json.load(f)

    num_right = 0
    num_error = 0
    for data in output_datas:
        answers = ground_truth_datas
        results = data['turbo_results']
        if check_string(results):
            response = clean_results(results)
            if response=="NULL":
                response = results
            else:
                if exact_match(response, answers):
                    num_right+=1
                else:
                    num_error+=1
        else:
            response = results
            if args.constraints_refuse and check_string(response):
                continue
            if exact_match(response, answers):
                num_right+=1
            else:
                num_error+=1

    print("Exact Match: {}".format(float(num_right/len(output_datas))))
    print("right: {}, error: {}".format(num_right, num_error))

    save_result2json(args.dataset, num_right, num_error, len(output_datas))
    