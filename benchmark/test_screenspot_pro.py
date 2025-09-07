import copy
import itertools
 
import torch
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import multiprocessing
logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)
 
GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'], help="Instruction style to use.")
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en', help="Language to use.")
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'], help="Ground truth type: 'positive' or 'negative'.")
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument("--planner", type=str, required=False, default="None")
    parser.add_argument("--num_crops", type=int, default=7)
    parser.add_argument("--output_format", type=str, default="point")
    parser.add_argument("--input_format", type=str, default="IF")  # natural / IF
 
    args = parser.parse_args()
    return args
 
def get_model(args):
    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        max_num_seqs=10,
        tensor_parallel_size=1
    )
    return llm
 
def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    """
    Filters the results based on provided values. None means include all (ignore filtering this attribute).
 
    Parameters:
        results (list): A list of dictionaries containing sample results.
   
    Returns:
        list: A filtered list of dictionaries based on the given criteria.
    """
    filtered_results = []
 
    for sample in results:
        # Check each filter condition; if None, consider it as passed
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)
 
    return filtered_results
 
 
def make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    """
    Returns a list of combinations of values for attributes where the corresponding parameter is set to True.
    """
    # Initialize a dictionary to store unique values for each attribute
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }
 
    # Collect unique values from the results
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))
 
    # Filter out the attributes that are set to False (no need for combinations)
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []
 
    # Generate all combinations of the selected attributes using itertools.product
    attribute_combinations = list(itertools.product(*filtered_values.values()))
 
    # Convert combinations into dictionaries with corresponding attribute names
    combinations = []
    for combination in attribute_combinations:
        combinations.append(dict(zip(filtered_values.keys(), combination)))
 
    return combinations
 
 
def calc_metric_for_result_list(results):
    """Calculates the metrics for a simple result list."""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")
    iou_5 = sum([res["IoU@0.5"] for res in results])
    iou = sum([res["IoU"] for res in results])
 
    # Calculate text and icon specific metrics using collect_results_to_eval
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")
 
    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0,
        "IoU@0.5": iou_5 / num_total if num_total > 0 else 0,
        "IoU": iou / num_total if num_total > 0 else 0
    }
    return metrics
 
 
def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
    # bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, w, h
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
   
    click_point = response["point"]  # may be none
    # print(click_point)
    if click_point is None:
        return "wrong_format"
    # Check if the predicted point falls in the ground truth box
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"
   
def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else: ## response["result"] == wrong_format
        return "wrong_format"
 
def evaluate_fine_grained(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results,
        platform=True,
        application=True,
        instruction_style=True,
        gt_type=True
    )
 
    evaluation_result = {}
 
    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
       
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            application=application,
            instruction_style=inst_style,
            gt_type=gt_type
        )
       
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
       
        # Construct a unique key based on the combination
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics
 
    return evaluation_result
 
def evaluate_seeclick_paper_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results,
        platform=True,
        instruction_style=True,
        gt_type=True
    )
 
    evaluation_result = {}
 
    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
       
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            instruction_style=inst_style,
            gt_type=gt_type
        )
       
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
       
        # Construct a unique key based on the combination
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics
 
    return evaluation_result
 
def evaluate_leaderboard_detailed_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results,
        application=True,
    )
 
    evaluation_result = {}
 
    # Iterate through each combination
    for combo in combinations:
        application = combo.get("application")
       
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            application=application,
        )
       
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
       
        # Construct a unique key based on the combination
        key = f"app:{application}"
        evaluation_result[key] = metrics
 
    return evaluation_result
 
def evaluate_leaderboard_simple_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results,
        group=True,
    )
 
    evaluation_result = {}
 
    # Iterate through each combination
    for combo in combinations:
        group = combo.get("group")
       
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            group=group,
        )
       
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
       
        # Construct a unique key based on the combination
        key = f"group:{group}"
        evaluation_result[key] = metrics
 
    return evaluation_result
 
def evaluate_overall(results):
    """
    Evaluates the overall metrics for all results without any filtering.
   
    Parameters:
        results (list): A list of dictionaries containing sample results.
       
    Returns:
        dict: A dictionary containing the overall metrics.
    """
    # Calculate metrics for the entire result set
    metrics = calc_metric_for_result_list(results)
   
    return metrics
 
 
def evaluate(results):
    """Collect results and calculate metrics. You can comment out function calls or add new ones based on your need.
    """
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }
 
    # TODO: comment out function calls based on your need
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = evaluate_overall(results)
    print(result_report["metrics"]["overall"])
 
    # Save detailed results
    result_report["details"] = results
 
    return result_report
 
 
def process_input(content, input_format):
    imgs = []
    texts = []
    for ct in content:
        if ct["type"] == "image":
            imgs.append(ct)
        elif ct["type"] == "text":
            texts.append(ct)
        else:
            raise RuntimeError(f"error data: {content}")
    if input_format == "IF":
        return texts + imgs
    elif input_format == "natural":
        return imgs + texts
    else:
        raise NotImplementedError
 
def process_image(img, bbox, num_crops):

    if num_crops == 16:
        target_width, target_height = 336 * 5, 336 *3
    elif num_crops == 7:
        target_width, target_height = 336 * 3, 336 *2
    elif num_crops == 28:
        target_width, target_height = 336 * 7, 336 *4
    else:
        raise NotImplementedError

    img_ratio = img.width / img.height  
    target_ratio = target_width / target_height
   
    if img_ratio > target_ratio:  
        new_width = target_width  
        new_height = int(new_width / img_ratio)
    else:  
        new_height = target_height
        new_width = int(new_height * img_ratio)  
    reshape_ratio = new_width / img.width
    new_bbox = [int(k*reshape_ratio) for k in bbox]
 
    img = img.resize((new_width, new_height), Image.LANCZOS)  
    new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))  
    paste_position = (0, 0)  
    new_img.paste(img, paste_position)
    return new_img, new_bbox, reshape_ratio
 

def get_acc(pdbox, gtbox):
    center_x = (pdbox[0] + pdbox[2])/2
    center_y = (pdbox[1] + pdbox[3])/2
    if center_x >= gtbox[0] and center_x <= gtbox[2] and center_y >= gtbox[1] and center_y <= gtbox[3]:
        return "correct"
    else:
        return "wrong"
 
def calculate_iou(box1, box2):  
    """  
    Calculate the Intersection over Union (IoU) of two bounding boxes.  
     
    Parameters:  
    box1 (tuple): A tuple (x1, y1, x2, y2) representing the first bounding box.  
    box2 (tuple): A tuple (x1, y1, x2, y2) representing the second bounding box.  
     
    Returns:  
    float: The IoU of the two bounding boxes.  
    """  
     
    x1_1, y1_1, x2_1, y2_1 = box1  
    x1_2, y1_2, x2_2, y2_2 = box2  
     
    # Calculate the (x, y)-coordinates of the intersection rectangle  
    xi1 = max(x1_1, x1_2)  
    yi1 = max(y1_1, y1_2)  
    xi2 = min(x2_1, x2_2)  
    yi2 = min(y2_1, y2_2)  
     
    # Calculate the area of the intersection rectangle  
    inter_width = max(0, xi2 - xi1)  
    inter_height = max(0, yi2 - yi1)  
    inter_area = inter_width * inter_height  
     
    # Calculate the area of both bounding boxes  
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)  
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)  
     
    # Calculate the union area  
    union_area = box1_area + box2_area - inter_area  
     
    # Calculate the IoU  
    iou = inter_area / union_area if union_area != 0 else 0  
     
    return iou 

def get_results(img_paths, groups, platforms, applications, langs, instruction_styles, prompt_to_evaluates,
                gt_types, ui_types, task_filenames, image_sizes, reshape_ratios, preds, bboxs, output_format):
    assert len(preds) == len(bboxs)
    results = []
 
    for (img_path, group, platform, application, lang, instruction_style, prompt_to_evaluate, gt_type, ui_type, task_filename,
        size, reshape_ratio, pred, bbox) in zip(img_paths, groups, platforms, applications, langs, instruction_styles,
        prompt_to_evaluates, gt_types, ui_types, task_filenames, image_sizes, reshape_ratios, preds, bboxs):
       
        try:
            if output_format=="xyxy":
                pred = pred.strip().split("<box>")[1].split("</box>")[0]
                coors = [float(o) for o in pred.split(", ")]
                assert len(coors) == 4
            elif output_format=="xywh":
                pred = pred.strip().split("<box>")[1].split("</box>")[0]
                coors = [float(o) for o in pred.split(", ")]
                assert len(coors) == 4
                coors = [coors[0], coors[1], coors[0]+coors[2], coors[1]+coors[3]]
            elif output_format=="point":
                pred = pred.strip().split("<point>")[1].split("</point>")[0]
                coors = [float(o) for o in pred.split(", ")]
                assert len(coors) == 2
                coors = [coors[0], coors[1], coors[0], coors[1]]
            elif output_format=="midwh":
                pred = pred.strip().split("<box>")[1].split("</box>")[0]
                coors = [float(o) for o in pred.split(", ")]
                assert len(coors) == 4
                coors = [coors[0]-coors[2]/2, coors[1]-coors[3]/2, coors[0]+coors[2]/2, coors[1]+coors[3]/2]
        except Exception as e:
            print("error parsing: "+str(e))
            continue
 
        if size is not None:
            coors[0] *= size[0] / 1000
            coors[1] *= size[1] / 1000
            coors[2] *= size[0] / 1000
            coors[3] *= size[1] / 1000
 
        correctness = get_acc(coors, bbox)
        iou_5 = 0
        iou = 0
        if not output_format == "point":
            iou = calculate_iou(coors, bbox)

            if iou >=0.5:
                iou_5 = 1

        sample_result = {
            "img_path": img_path,
            "group": group,
            "platform": platform,
            "application": application,
            "lang": lang,
            "instruction_style": instruction_style,
            "prompt_to_evaluate": prompt_to_evaluate,
            "gt_type": gt_type,
            "ui_type": ui_type,
            "task_filename": task_filename,
            "reshape_ratio": reshape_ratio,
            "pred": coors,
            "bbox": bbox,
            "correctness": correctness,
            "IoU@0.5": iou_5,
            "IoU": iou
        }
        results.append(sample_result)
       
    return results
 

def process_task(inp):  
    sample, args = inp
    try:  
        result = {}  
        filename = sample["img_filename"]  
        img_path = os.path.join(args.screenspot_imgs, filename)  
        re = "\nThe description of the element: \n"

        if args.planner in ["gpt-4o", "o4-mini"]:  
            long = sample[args.planner]["Output"] # if "4o" in args.screenspot_test else sample["re"]["Output"] 
            func = long["functional_reference"]  
            pos = long["positional_reference"]  
            appear = long["appearance_reference"]  
            re += func  
            re += "\n"  
            re += pos  
            re += "\n"  
            re += appear  
        else:  
            if sample['instruction'] is not None:  
                instruction = sample['instruction']  
                re += f"{instruction}\n"  
          
        re += f"\n\nLocate the above described element in the image. The output should be bounding box using relative coordinates multiplying 1000.\n"
        ctns = [{"type": "text", "text": re}]  
        ctns.append({"type": "image", "path": img_path})  
        ctns = process_input(ctns, args.input_format)  
        img = Image.open(img_path)  
        gt = sample["bbox"]  
  
        img, gt, reshape_ratio = process_image(img, gt, args.num_crops)  
        w, h = img.size  
  
        image_list = []  
        img_id = 1  
        prompt = "<|user|> \n"  
        for ct in ctns:  
            if ct["type"] == "text":  
                prompt += ct["text"]  
            else:  
                prompt += f"<|image_{img_id}|> \n"  
                img_id += 1  
                image_list.append(img)  
          
        prompt += "<|end|> \n<|assistant|>" 
        result["prompt_data"] = {  
            "prompt": prompt,  
            "multi_modal_data": {  
                "image": image_list  
            }  
        }  
  
        group = sample["group"] if "group" in sample else None  
        result["img_path"] = img_path  
        result["group"] = group  
        result["platform"] = sample["platform"]  
        result["application"] = sample["application"]  
        result["language"] = sample["language"]  
        result["instruction_style"] = sample["instruction_style"]  
        result["prompt_to_evaluate"] = sample["prompt_to_evaluate"]  
        result["gt_type"] = sample["gt_type"]  
        result["ui_type"] = sample["ui_type"]  
        result["task_filename"] = sample["task_filename"]  
        result["image_size"] = (w, h)  
        result["bbox"] = gt  
        result["reshape_ratio"] = reshape_ratio  
  
        return result  
  
    except Exception as e:  
        print(e)  
        print("wrong format")  
        return None  

 
def main(args): 
    llm = get_model(args)
    print("Load model success")
 
    if args.task == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ]
    else:
        task_filenames = args.task.split(",")
 
    if args.inst_style == "all":
        inst_styles = INSTRUCTION_STYLES
    else:
        inst_styles = args.inst_style.split(",")
 
    if args.language == "all":
        languages = LANGUAGES
    else:
        languages = args.language.split(",")
 
    if args.gt_type == "all":
        gt_types = GT_TYPES
    else:
        gt_types = args.gt_type.split(",")

    tasks_to_run = []
    for task_filename in task_filenames:
        dataset = task_filename + ".json"
        with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
            task_data = json.load(f)
 
        # Create the list of tasks to run, one item as an instance. Tasks may be reused.
        for inst_style in inst_styles:  # Expand tasks based on user configurations
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        task_instance = copy.deepcopy(task_instance)
                        task_instance["task_filename"] = task_filename
                        task_instance["gt_type"] = gt_type
                        task_instance["instruction_style"] = inst_style
                        task_instance["language"] = lang
                        if lang == "cn":
                            if inst_style!= 'instruction' or gt_type != 'positive':
                                # TODO: Translate the data
                                raise AttributeError("Only positive samples and 'instruction' style are supported for Chinese instructions.")
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        elif lang == "en":
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]
 
                        tasks_to_run.append(task_instance)
        print(f"Num of sample in {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} = {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    
    # tasks_to_run = tasks_to_run[:10]
    print(f"Total tasks: {len(tasks_to_run)}")
 
    prompt_list = []
 
    img_paths = []
    groups = []
    platforms = []
    applications = []
    langs = []
    instruction_styles = []
    prompt_to_evaluates = []
    gt_types = []
    ui_types = []
    task_filenames = []
    image_sizes = []
    bboxs = []
    reshape_ratios = []


    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  
    results = list(tqdm(pool.imap(process_task, [(k, args) for k in tasks_to_run]), total=len(tasks_to_run)))  
    for result in results:  
        if result is not None:  
            prompt_list.append(result["prompt_data"])  
            img_paths.append(result["img_path"])  
            groups.append(result["group"])  
            platforms.append(result["platform"])  
            applications.append(result["application"])  
            langs.append(result["language"])  
            instruction_styles.append(result["instruction_style"])  
            prompt_to_evaluates.append(result["prompt_to_evaluate"])  
            gt_types.append(result["gt_type"])  
            ui_types.append(result["ui_type"])  
            task_filenames.append(result["task_filename"])  
            image_sizes.append(result["image_size"])  
            bboxs.append(result["bbox"])  
            reshape_ratios.append(result["reshape_ratio"])
        else:
            assert False  
    pool.close()  
    pool.join()

    
   
    outputs = llm.generate(prompt_list, sampling_params = SamplingParams(temperature=0, max_tokens=64))   # 32007, 32000
     
    preds = [o.outputs[0].text for o in outputs]
    # print(preds)

 
    results = get_results(img_paths, groups, platforms, applications, langs, instruction_styles, prompt_to_evaluates,
                            gt_types, ui_types, task_filenames, image_sizes, reshape_ratios, preds, bboxs, output_format = args.output_format)
 
    result_report = evaluate(results)
    # Save to file
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(result_report, f, indent=4)
    logging.info("Evaluation of ScreenSpotPro finished.")

 
if __name__ == "__main__":
    main(parse_args())