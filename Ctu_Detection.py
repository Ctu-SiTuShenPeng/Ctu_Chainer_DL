# -*- coding: utf-8 -*-
import Lib.tools.Ctu_Detection import Ctu_Detection,Ctu_Detection_Predictor,Ctu_Detection_Predictor_Work

if __name__ == '__main__':
    func_type = 'predict'
    if func_type=='train':
        work_thread='0'
        # 训练
        ctu_train = Ctu_Detection(
            use_gpu='0',min_size=600,max_size=1000,mean = [123.15163084, 115.90288257, 103.0626238],
                 lic_file='license.lic', time_sleep=5, log_flag='1', thread_check='1', 
                 project_name='ctu_det',result_model='ctu_result_model'
        )
        for each_key in ctu_train.func_signal.keys():
            ctu_train.connect_signal(each_key,get_recv_data_det_tools)
        init_res = ctu_train.init_model(data_dir=r'dataset/DataSet_Tablet',train_split=0.9,batch_size=4,model_name='fasterrcnn_resnet18',alpha=0.25,pre_model=None,aux=None,n_processes=None)
        if init_res['return_value']=='1':
            ctu_train.train(train_num=1, learning_rate=None, optimizer_type="adam", result_model='temp_result',work_thread=work_thread)
            if work_thread=='1':
                print('等待训练线程启动')
                time.sleep(5)
                while ctu_train.train_flag:
                    time.sleep(0.5)
        ctu_train.shutdown_object()
        del ctu_train
    elif func_type == 'predict':
        simple = False
        ctu_predict = Ctu_Detection(use_gpu='0',lic_file="license.lic", time_sleep=5, log_flag='1', thread_check='1',project_name='ctu_det')
        for each_key in ctu_predict.func_signal.keys():
            ctu_predict.connect_signal(each_key,get_recv_data_det_tools)
        load_res = ctu_predict.load_model(params_file='ctu_result_model/ctu_det/ctu_params.json', use_best=True, snid='')
        if load_res['return_value']=='1':
            data_list = []
            predict_cvs=[]
            predictNum=2
            cv2.namedWindow("result", 0)
            cv2.resizeWindow("result", 640, 480)
            for root, dirs, files in os.walk(r'dataset/DataSet_Tablet/test'):
                for f in files:
                    data_list.append(os.path.join(root, f))
            data_list_split = [data_list[i:i+predictNum] for i in range(0,len(data_list),predictNum)]
            for each_data in data_list_split:
                predict_cvs.clear()
                for each_file in each_data:
                    img_cv = read_image(each_file)
                    if img_cv is None:
                        continue
                    predict_cvs.append(img_cv)
                res_list = ctu_predict.predict(predict_cvs, snid='',simple='1' if simple==True else '0',predict_score=0.6)
                print(f"耗时:{res_list['time']} ms")
                for each_id in range(0,len(res_list['predict_output'])):
                    for each_bbox in res_list['predict_output'][each_id]['bbox']:
                        print(each_bbox)
                    cv2.imshow("result", res_list['predict_output'][each_id]['image_result'])
                    cv2.waitKey()
            cv2.destroyWindow('result')
        ctu_predict.shutdown_object()
        del ctu_predict
    elif func_type == 'get_fps':
        ctu_fps = Ctu_Detection(use_gpu='0',lic_file="license.lic", time_sleep=5, log_flag='1', thread_check='1',project_name='ctu_det')
        for each_key in ctu_fps.func_signal.keys():
            ctu_fps.connect_signal(each_key,get_recv_data_det_tools)
        load_res = ctu_fps.load_model(params_file='ctu_result_model/ctu_det/ctu_params.json', use_best=True, snid='')
        if load_res['return_value']=='1':
            result_mes = ctu_fps.get_fps(batch_size=2, test_interval=10, snid='')
            print(result_mes['message'])
        ctu_fps.shutdown_object()
        del ctu_fps
    elif func_type=='get_onnx':
        ctu_onnx = Ctu_Detection(use_gpu='0',lic_file="license.lic", time_sleep=5, log_flag='1', thread_check='1',project_name='ctu_cls')
        for each_key in ctu_onnx.func_signal.keys():
            ctu_onnx.connect_signal(each_key,get_recv_data_det_tools)
        load_res = ctu_onnx.load_model(params_file='ctu_model/det/ctu_det_centernet_hourglassnet/ctu_params.json', use_best=True, snid='')
        if load_res['return_value']=='1':
            ctu_onnx.convert_onnx("ctu_model/cls/ctu_cls_chess/resnet18.onnx")
        ctu_onnx.shutdown_object()
        del ctu_onnx
    elif func_type=='predictor':
        simple = False
        predict_score = 0.5
        ctu_predictor=Ctu_Detection_Predictor(use_gpu='0', lic_file='license.lic', time_sleep=5, log_flag='1', thread_check='1', project_name='ctu_det',result_model='ctu_logs',func=get_recv_data_det_tools)
        load_res=ctu_predictor.load_model(params_file='ctu_result_model/ctu_det/ctu_params.json', use_best=True)
        if load_res['return_value']=='1':
            data_list = []
            predict_cvs=[]
            predictNum=1
            cv2.namedWindow("result", 0)
            cv2.resizeWindow("result", 640, 480)
            for root, dirs, files in os.walk(r'dataset/DataSet_Tablet/test'):
                for f in files:
                    data_list.append(os.path.join(root, f))
            data_list_split = [data_list[i:i+predictNum] for i in range(0,len(data_list),predictNum)]
            for each_data in data_list_split:
                predict_cvs.clear()
                for each_file in each_data:
                    img_cv = read_image(each_file)
                    if img_cv is None:
                        continue
                    predict_cvs.append(img_cv)
                res_list = ctu_predictor.predict(predict_cvs, snid='',simple='1' if simple==True else '0',predict_score=0.5)
                print(res_list)
                print(f"耗时:{res_list['time']} ms")
                for each_id in range(0,len(res_list['predict_output'])):
                    for each_bbox in res_list['predict_output'][each_id]['bbox']:
                        print(each_bbox)
                    cv2.imshow("result", res_list['predict_output'][each_id]['image_result'])
                    cv2.waitKey()
            cv2.destroyWindow('result')
        ctu_predictor.del_thread()
        del ctu_predictor
    else:
        simple = False
        run_model='None'
        pool_dict={
            'thread_pool':{
                "run_mode": "thread_pool",
                "thread_num": 4,
                "result_expire_seconds": 600
            },
            'thread_queue':{
                "run_mode": "thread_queue",
                "max_queue_size": 1000,
                "result_expire_seconds": 600
            },
            'process_pool':{
                "run_mode": "process_pool",
                "thread_num": 2,  # 进程数
                "max_queue_size": 0,
                "result_expire_seconds": 600
            },
            'None':None
       }
                        
        img_file=r'dataset/DataSet_Tablet/test'

        ctu_model = Ctu_Detection_Predictor_Work(
            use_gpu = '0',
            project_name = 'ctu_det',
            pool_dict = pool_dict[run_model],
            lic_file = 'license.lic',
            func_pool = get_recv_data_det_tools
        )
        # 加载模型
        res_load = ctu_model.load_model('ctu_result_model/ctu_det/ctu_params.json')
        print(f"{res_load['message']}")
        if res_load['return_value'] == '1':
            image_list = []
            for root, dirs, files in os.walk(img_file):
                for f in files:
                    image_list.append(os.path.join(root, f))
            
            if pool_dict[run_model] is not None:
                task_snids=[]
                for each_index in range(0,len(image_list)):
                    img_cv = cv2.imread(image_list[each_index],1)
                    snid = f"task_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{each_index+1}"
                    
                    submit_res = ctu_model.submit_predict_task(img_cv=img_cv,snid=snid, simple='0' if simple==False else '1',predict_score=0.5)
                    if submit_res['return_value'] == '1':
                        task_snids.append(snid)
                        print(f"提交成功 snid={snid}")
                    else:
                        print(f"提交失败 snid={snid}: {submit_res['message']}")
                
                # time.sleep(120)
                cv2.namedWindow("result_pool", 0)
                cv2.resizeWindow("result_pool", 640, 480)
                for snid in task_snids:
                    # 循环查询直到任务完成
                    while True:
                        res = ctu_model.get_task_status(snid)
                        print(res['status'])
                        if res['status'] in ['success', 'failed']:
                            print(f"snid={snid} 结果: {res['status']} | {res['message']}")
                            if res['status'] == 'success':
                                res_list = res['return_data']
                                print(f"耗时:{res_list['time']} ms")
                                for each_id in range(0,len(res_list['predict_output'])):
                                    for each_bbox in res_list['predict_output'][each_id]['bbox']:
                                        print(each_bbox)
                                    cv2.imshow("result_pool", res_list['predict_output'][each_id]['image_result'])
                                    cv2.waitKey()
                            break
                        time.sleep(0.001)  # 短暂等待   
                cv2.destroyWindow("result_pool")    

            else:
                data_list = []
                predict_cvs=[]
                predictNum=1
                cv2.namedWindow("result", 0)
                cv2.resizeWindow("result", 640, 480)
                for root, dirs, files in os.walk(img_file):
                    for f in files:
                        data_list.append(os.path.join(root, f))
                data_list_split = [data_list[i:i+predictNum] for i in range(0,len(data_list),predictNum)]
                for each_data in data_list_split:
                    predict_cvs.clear()
                    for each_file in each_data:
                        img_cv = read_image(each_file)
                        if img_cv is None:
                            continue
                        predict_cvs.append(img_cv)
                    res_list = ctu_model.submit_predict_task(predict_cvs, snid=get_snid(),simple='0' if simple==False else '1',predict_score=0.5)
                    print(f"耗时:{res_list['time']} ms")
                    for each_id in range(0,len(res_list['predict_output'])):
                        for each_bbox in res_list['predict_output'][each_id]['bbox']:
                            print(each_bbox)
                        cv2.imshow("result", res_list['predict_output'][each_id]['image_result'])
                        cv2.waitKey()
                cv2.destroyWindow("result")    
            ctu_model.shutdown()
            del ctu_model
