import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())
sys.path.append("../..")

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive

import json

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    #parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device", type=str, default="0")
#     parser.add_argument("--model_file", type=str, default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")
    parser.add_argument("--category", type=str, default="all", help = "all, oEffect, oReact, oWant, xAttr, xEffect, xIntent, xNeed, xReact, xWant")
    parser.add_argument("--model_file", type=str, default="../../"+"pretrained_models/atomic_pretrained_model.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="beam-5", help = "greedy/beam-# where # is the beam size/topk-# where # is k")
    parser.add_argument("--input_file", type=str, default="../../Dataset/DailyDialog/DailyDialog_json.json")
    parser.add_argument("--output_file", type=str, default="../../Dataset/DailyDialog/DailyDialog_comet_10000_20000.json")


    args = parser.parse_args()

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"


    sampling_algorithm = args.sampling_algorithm
    #category = args.category
    #args.category = ["xAttr", "xReact", "oReact"]
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    list_of_input_events = ["I will go to school! I will go to school! I will go to school! I will go to school! I will go to school! I will go to school!",
                     "good",
                     "beautiful campus",]
    list_of_input_events = []
    with open(args.input_file, encoding='utf-8') as f:
        corpus = json.load(f)
    for dataname,data in corpus.items():
        print(dataname)
        for i,d in enumerate(data):

            #if len(set(d["labels"])) == 1 : continue
            # utterances = d["utterances"]
            # for u in utterances:
            #     list_of_input_events.append(u)
            
            if len(set(d["labels"])) == 1:
                utterances = d["utterances"]
                for u in utterances:
                    list_of_input_events.append(u)
    
    news_list_of_input_events = []
    for i in list_of_input_events:
        if i not in news_list_of_input_events:
            news_list_of_input_events.append(i)
    list_of_input_events = news_list_of_input_events

    #list_of_input_events = list(set(list_of_input_events))
    list_of_input_events = list_of_input_events[10000:20000]
    
    print(len(list_of_input_events))
    print(list_of_input_events[:5])


    output_file = open(args.output_file, 'a', encoding = 'utf-8')

    for idx, sentence in enumerate(list_of_input_events):

        outputs = interactive.get_atomic_sequence(
            "PersonX says " + sentence, model, sampler, data_loader, text_encoder, args.category)
        if idx%100 ==0:
            print(idx)
        #print(outputs)
        output_file.write(str(outputs))
        output_file.write("\n")
        output_file.flush()
    output_file.close()
    # print(type(outputs))



    # with open(args.output_file, 'w', encoding = 'utf-8') as f:
    #     json.dump(corpus,f,sort_keys=False, indent=4, separators=(',', ': '), ensure_ascii=False)










