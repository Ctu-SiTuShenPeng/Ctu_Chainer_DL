from ctu_chainer import ctu_dl_api_full,ctu_api_predictor,ctu_api_traintor
def get_return_func(sig,mes):
    # print('1',sig,mes)
    # if 'predict' in sig:
    #     print('1',sig,mes)
    pass

if __name__ == "__main__":
    run_func = 'server'
    if run_func=='dl_api_full':
        ctu = ctu_dl_api_full(func_tool=get_return_func,lic_file='license.lic',result_model = 'ctu_result_model')
        # 创建模型
        if False:
            res_inif = ctu.create_model(use_gpu = '0', model_type='cls',project_name='ctu_cls',image_size=224,
                            data_dir=r'dataset/DataSet_Chess/DataImage',train_split=0.9,batch_size=4,model_name='resnet18',alpha=0.25,pre_model=None,aux=None,n_processes=None,
                            train_num=2, learning_rate=None, optimizer_type="adam")
            res_inif = ctu.create_model(use_gpu = '0', model_type='det',project_name='ctu_det',min_size=600,max_size=1000,
                            data_dir=r'dataset/DataSet_Tablet', train_split=0.9, batch_size=1, model_name='fasterrcnn_resnet18', alpha=0.25, pre_model=None, aux=None,n_processes=None,
                            train_num=2, learning_rate=None, optimizer_type="adam")
            res_inif = ctu.create_model(use_gpu = '0', model_type='seg',project_name='ctu_seg',image_size=512,
                            data_dir=r'dataset/DataSet_Tablet/DataJson',train_split=0.9,batch_size = 1, model_name="deeplab_resnet18", alpha=0.25, pre_model=None,encoding='gbk',aux=None,n_processes=None,
                            train_num=2, learning_rate=None, optimizer_type="adam")
            res_inif = ctu.create_model(use_gpu = '0', model_type='ins',project_name='ctu_ins',min_size=600,max_size=1000,
                            data_dir=r'dataset/DataSet_Tablet/DataJson',train_split=0.9,batch_size=1,model_name='maskrcnn_resnet18',alpha=0.25,pre_model=None,aux=None,encoding='gbk',n_processes=None,
                            train_num=2, learning_rate=None, optimizer_type="adam")
        # 预测
        if False:
            # # cls
            # img_file=r'dataset/DataSet_Chess/test'
            # image_list=[]
            # simple=True
            # for root, dirs, files in os.walk(img_file):
            #     for f in files:
            #         image_list.append(os.path.join(root, f))
            # cv2.namedWindow("result", 0)
            # cv2.resizeWindow("result", 640, 480)
            # for each_index in range(0,len(image_list)):
            #     img_cv = read_image(image_list[each_index])
            #     snid_mes = f"task_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{each_index+1}"
            #     submit_res = ctu.predict(project_name='ctu_cls',img_cv=img_cv,snid=snid_mes,simple=simple)
            #     if submit_res['return_value']=='1':
            #         if 'status' not in list(submit_res.keys()) :
            #             print(f"耗时:{submit_res['time']} ms")
            #             for each_index in range(len(submit_res['predict_output'])):
            #                 print(submit_res['predict_output'][each_index]['classes_names'],submit_res['predict_output'][each_index]['score'])
            #                 if simple==False:
            #                     cv2.imshow("result", submit_res['predict_output'][each_index]['image_base'])
            #                     cv2.waitKey()
            #                 else:
            #                     print(submit_res)
            #         else:
            #             print(f'等待回调函数:snid={snid_mes}')
            #     else:
            #         print('预测异常:{0}'.format(submit_res['message']))
            # cv2.destroyWindow('result')

            # det
            img_file=r'dataset/DataSet_Tablet/test'
            image_list=[]
            simple=True
            for root, dirs, files in os.walk(img_file):
                for f in files:
                    image_list.append(os.path.join(root, f))
            cv2.namedWindow("result", 0)
            cv2.resizeWindow("result", 640, 480)
            for each_index in range(0,len(image_list)):
                img_cv = read_image(image_list[each_index])
                snid_mes = f"task_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{each_index+1}"
                submit_res = ctu.predict(project_name='ctu_det',img_cv=img_cv,snid=snid_mes,simple=simple,predict_score=0.5)
                if submit_res['return_value']=='1':
                    if 'status' not in list(submit_res.keys()) :
                        print(f"耗时:{submit_res['time']} ms")
                        for each_id in range(0,len(submit_res['predict_output'])):
                            for each_bbox in submit_res['predict_output'][each_id]['bbox']:
                                print(each_bbox)
                            if simple==False:
                                cv2.imshow("result", submit_res['predict_output'][each_id]['image_result'])
                                cv2.waitKey()
                            else:
                                print(submit_res)
                    else:
                        print(f'等待回调函数:snid={snid_mes}')
                else:
                    print('预测异常:{0}'.format(submit_res['message']))
            cv2.destroyWindow('result')

            # # seg
            # img_file=r'dataset/DataSet_Tablet/test'
            # image_list=[]
            # simple=True
            # for root, dirs, files in os.walk(img_file):
            #     for f in files:
            #         image_list.append(os.path.join(root, f))
            # cv2.namedWindow("result", 0)
            # cv2.resizeWindow("result", 640, 480)
            # for each_index in range(0,len(image_list)):
            #     img_cv = read_image(image_list[each_index])
            #     snid_mes = f"task_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{each_index+1}"
            #     submit_res = ctu.predict(project_name='ctu_seg',img_cv=img_cv,snid=snid_mes,simple=simple)
            #     if submit_res['return_value']=='1':
            #         if 'status' not in list(submit_res.keys()) :
            #             print(f"耗时:{submit_res['time']} ms")
            #             for each_id in range(0,len(submit_res['predict_output'])):
            #                 for each_id in range(0,len(submit_res['predict_output'])):
            #                     if simple==False:
            #                         cv2.imshow("result", submit_res['predict_output'][each_id]['image_result'])
            #                         cv2.waitKey()
            #                     else:
            #                         cv2.imshow("result", submit_res['predict_output'][each_id]['image_label'])
            #                         cv2.waitKey()
            #         else:
            #             print(f'等待回调函数:snid={snid_mes}')
            #     else:
            #         print('预测异常:{0}'.format(submit_res['message']))
            # cv2.destroyWindow('result')

            # # ins
            # img_file=r'dataset/DataSet_Tablet/test'
            # image_list=[]
            # simple=True
            # for root, dirs, files in os.walk(img_file):
            #     for f in files:
            #         image_list.append(os.path.join(root, f))
            # cv2.namedWindow("result", 0)
            # cv2.resizeWindow("result", 640, 480)
            # for each_index in range(0,len(image_list)):
            #     img_cv = read_image(image_list[each_index])
            #     snid_mes = f"task_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{each_index+1}"
            #     submit_res = ctu.predict(project_name='ctu_ins',img_cv=img_cv,snid=snid_mes,simple=simple,predict_score=0.5)
            #     if submit_res['return_value']=='1':
            #         if 'status' not in list(submit_res.keys()) :
            #             print(f"耗时:{submit_res['time']} ms")
            #             for each_id in range(0,len(submit_res['predict_output'])):
            #                 for each_bbox in submit_res['predict_output'][each_id]['target_list']:
            #                     print(each_bbox)
            #                 if simple==False:
            #                     cv2.imshow("result", submit_res['predict_output'][each_id]['image_result'])
            #                     cv2.waitKey()
            #                 else:
            #                     print(submit_res)
            #         else:
            #             print(f'等待回调函数:snid={snid_mes}')
            #     else:
            #         print('预测异常:{0}'.format(submit_res['message']))
            # cv2.destroyWindow('result')
        # http
        if True:
            try:
                header = {"Content-Type": "application/json;charset=UTF-8"}
                postData = {
                    'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    'snid':'a76543dgfdsacc',
                    'project_name':'ctu_ins',
                    'img_cv':image_to_base64(read_image('dataset/DataSet_Tablet/DataImage/pill_bag_001.png')),
                    'simple':'0',
                    'predict_score':0.5,
                    'get_count':5
                }
                res_json = json.loads(requests.post(url='http://127.0.0.1:54321/predict', data=json.dumps(postData),headers=header).text)
                print("请求正常:",res_json)     
            except Exception as e:
                print("请求异常:",str(e))

        while True:
            # time.sleep(10)
            ctu.shutdown_object()
            break
        del ctu
    elif run_func=='dl_api_predictor':
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
                        
        img_file=r'dataset/DataSet_Chess/test'

        ctu_model = ctu_api_predictor(
            use_gpu = '0',
            project_name = 'ctu_cls',
            pool_dict = pool_dict[run_model],
            lic_file = 'license.lic',
            func_tool = get_return_func
        )
        # 加载模型
        res_load = ctu_model.load_model('ctu_result_model/ctu_cls/ctu_params.json',model_type='cls')
        print(f"{res_load['message']}")
        if res_load['return_value'] == '1':
            image_list = []
            for root, dirs, files in os.walk(img_file):
                for f in files:
                    image_list.append(os.path.join(root, f))
            
            if pool_dict[run_model]  is not None:
                task_snids=[]
                for each_index in range(0,len(image_list)):
                    img_cv = cv2.imread(image_list[each_index],1)
                    snid = f"task_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{each_index+1}"
                    
                    submit_res = ctu_model.submit_predict_task(img_cv=img_cv,snid=snid, simple='0' if simple==False else '1')
                    if submit_res['return_value'] == '1':
                        task_snids.append(snid)
                        print(f"提交成功 snid={snid}")
                    else:
                        print(f"提交失败 snid={snid}: {submit_res['message']}")
                    time.sleep(0.1)
                
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
                                for each_index in range(len(res_list['predict_output'])):
                                    print(res_list['predict_output'][each_index]['classes_names'],res_list['predict_output'][each_index]['score'])
                                    if 'image_base' in res_list['predict_output'][each_index].keys():
                                        cv2.imshow("result_pool", res_list['predict_output'][each_index]['image_base'])
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
                for root, dirs, files in os.walk(r'dataset/DataSet_Chess/test'):
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
                    res_list = ctu_model.submit_predict_task(predict_cvs, snid=get_snid(),simple='0' if simple==False else '1')
                    print(f"耗时:{res_list['time']} ms")
                    for each_index in range(len(predict_cvs)):
                        print(res_list['predict_output'][each_index]['classes_names'],res_list['predict_output'][each_index]['score'])
                        cv2.imshow("result", predict_cvs[each_index])
                        cv2.waitKey()
            
            ctu_model.shutdown()
            del ctu_model
    elif run_func == 'server_predict':
        # 接口名称：http://127.0.0.1:54321/check_heartbeat
        # 发送：两个字段：request_time:请求时间    snid:唯一标识符
        # 返回：message、return_value、sind、response_time

        # 载入模型:http://127.0.0.1:54321/load_model
        # 发送：四个字段：request_time:请求时间    snid:唯一标识符     model_type：类型，脏污是seg    params_file：配置文件
        # 返回：message、return_value、sind、response_time、reponse_data

        # 预测模型：http://127.0.0.1:54321/predict
        # 发送：四个字段：request_time:请求时间    snid:唯一标识符     img_cv：base64图像   simple：是否简化返回     predict_score:置信度
        # 返回：res_json["output_data"]["predict_output"][0]["target_list"]    点数据
        #      res_json["output_data"]["predict_output"][0]["image_result"]    base64图像数据

        # 预测模型：http://127.0.0.1:54321/predict_more
        # 发送：四个字段：request_time:请求时间    snid:唯一标识符     img_data：[base64图像]   simple：是否简化返回     predict_score:置信度
        # 返回：res_json["output_data"]["predict_output"][0]["target_list"]    点数据
        #      res_json["output_data"]["predict_output"][0]["image_result"]    base64图像数据

        ctu_model = ctu_api_predictor(
            use_gpu = '0',
            project_name = 'ctu_seg',
            pool_dict = None,
            lic_file = 'license.lic',
            func_tool = get_return_func,
            server_port=54321
        )
        
        if False:
            # 检测心跳
            while True:
                try:
                    header = {"Content-Type": "application/json;charset=UTF-8"}
                    postData = {
                        'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        'snid':'a76543dgfdsacc',
                    }
                    res_json = json.loads(requests.post(url='http://127.0.0.1:54321/check_heartbeat', data=json.dumps(postData),headers=header).text)
                    print("请求正常:",res_json)     
                    if res_json['return_value']=='1':
                        break
                except Exception as e:
                    print("请求异常:",str(e))
                
                time.sleep(1)
                
            # 获取允许模型列表（基本不用）
            while True:
                try:
                    header = {"Content-Type": "application/json;charset=UTF-8"}
                    postData = {
                        'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        'snid':'a76543dgfdsacc',
                        'model_type':'seg'
                    }
                    res_json = json.loads(requests.post(url='http://127.0.0.1:54321/get_model_list', data=json.dumps(postData),headers=header).text)
                    print("请求正常:",res_json)     
                    if res_json['return_value']=='1':
                        break
                except Exception as e:
                    print("请求异常:",str(e))
                
                time.sleep(1)
                
            # 载入模型
            while True:
                try:
                    header = {"Content-Type": "application/json;charset=UTF-8"}
                    postData = {
                        'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        'snid':'a76543dgfdsacc',
                        'model_type':'seg',
                        'params_file':'ctu_result_model/ctu_seg/ctu_params.json'
                    }
                    res_json = json.loads(requests.post(url='http://127.0.0.1:54321/load_model', data=json.dumps(postData),headers=header).text)
                    print("请求正常:",res_json)     
                    if res_json['return_value']=='1':
                        break
                except Exception as e:
                    print("请求异常:",str(e))
                
                time.sleep(1)
            
            
            # # 预测
            # img_file=r'dataset/DataSet_Tablet/DataImage'
            # image_list = []
            # for root, dirs, files in os.walk(img_file):
            #    for f in files:
            #        image_list.append(os.path.join(root, f))
            
            # if simple == '0':
            #    cv2.namedWindow("result", 0)
            #    cv2.resizeWindow("result", 640, 480)
            # for each_index in range(0,len(image_list)):
            #    while True:
            #        task_snids = f"task_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}_{each_index+1}"
            #        try:
            #            header = {"Content-Type": "application/json;charset=UTF-8"}
            #            postData = {
            #                'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
            #                'snid':task_snids,
            #                'img_cv':image_to_base64(read_image(image_list[each_index])),
            #                'simple':simple,
            #                'predict_score':0.9
            #            }
            #            res_json = json.loads(requests.post(url='http://127.0.0.1:54321/predict', data=json.dumps(postData),headers=header).text)
            #            # print("请求正常:",res_json)     
            #            if res_json['return_value']=='1' and res_json["output_data"]['return_value']=='1':
            #                if simple=='1':
            #                    print(res_json["output_data"]["predict_output"][0]["target_list"])
            #                else:
            #                    print(res_json["output_data"]["predict_output"][0]["target_list"])
            #                    cv2.imshow('result',base64_to_image(res_json["output_data"]["predict_output"][0]["image_result"]))
            #                    cv2.waitKey(0)
            #                break
            #            else:
            #                print('异常:{0}/{1}'.format(res_json['message'],res_json["output_data"]['message']))
            #        except Exception as e:
            #            print("请求异常:",str(e))
            #        time.sleep(1)
            # if simple == '0':
            #    cv2.destroyWindow("result")  



            # 预测 predict_more
            img_file=r'dataset/DataSet_Tablet/DataImage'
            image_list = []
            for root, dirs, files in os.walk(img_file):
                for f in files:
                    image_list.append(os.path.join(root, f))
            predict_num = 4
            if simple == '0':
                for each_index in (range(predict_num)):
                    cv2.namedWindow(f"result{each_index+1}", 0)
                    cv2.resizeWindow(f"result{each_index+1}", 640, 480)

            for each_index in range(0,len(image_list),3):
                while True:
                    task_snids = f"task_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}_{each_index+1}"
                    try:
                        header = {"Content-Type": "application/json;charset=UTF-8"}
                        postData = {
                            'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                            'snid':task_snids,
                            'img_data':[image_to_base64(read_image(image_list[each_id])) for each_id in range(each_index,each_index+predict_num)],
                            'simple':simple,
                            'predict_score':0.9
                        }
                        res_json = json.loads(requests.post(url='http://127.0.0.1:54321/predict_more', data=json.dumps(postData),headers=header).text)
                        print("请求正常:",res_json.keys())     
                        if res_json['return_value']=='1':
                                for each_num in range(predict_num):
                                    print(res_json["output_data"][each_num]["predict_output"][0]["target_list"])
                                if postData['simple']=='0':
                                    for each_num in range(predict_num):
                                        cv2.imshow(f"result{each_num+1}",base64_to_image(res_json["output_data"][each_num]["predict_output"][0]["image_result"]))
                                    cv2.waitKey(0)
                                break
                        else:
                            print('异常:{0}'.format(res_json['message']))
                    except Exception as e:
                        print("请求异常:",str(e))
                    time.sleep(1)
            if simple == '0':
                for each_index in (range(predict_num)):
                    cv2.destroyWindow(f"result{each_index+1}")  
            
        while True:
            time.sleep(0.05)
    elif run_func=='server_train':
        ctu = ctu_api_traintor(lic_file='license.lic',result_model = 'ctu_result_model_train',server_host='0.0.0.0', server_port=12345)

        if True:
            # 检测心跳
            while True:
                try:
                    header = {"Content-Type": "application/json;charset=UTF-8"}
                    postData = {
                        'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        'snid':'a76543dgfdsacc',
                    }
                    res_json = json.loads(requests.post(url='http://127.0.0.1:12345/check_heartbeat', data=json.dumps(postData),headers=header).text)
                    print("请求正常:",res_json)     
                    if res_json['return_value']=='1':
                        break
                except Exception as e:
                    print("请求异常:",str(e))
                
                time.sleep(1)
            
            # 获取允许模型列表（基本不用）
            while True:
                try:
                    header = {"Content-Type": "application/json;charset=UTF-8"}
                    postData = {
                        'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        'snid':'a76543dgfdsacc',
                        'model_type':'seg'
                    }
                    res_json = json.loads(requests.post(url='http://127.0.0.1:12345/get_model_list', data=json.dumps(postData),headers=header).text)
                    print("请求正常:",res_json)     
                    if res_json['return_value']=='1':
                        break
                except Exception as e:
                    print("请求异常:",str(e))
                
                time.sleep(1)
            
            # 训练
            while True:
                try:
                    header = {"Content-Type": "application/json;charset=UTF-8"}
                    postData = {
                        'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        'snid':'a76543dgfdsacc',
                        'use_gpu':'0',
                        'model_type':'seg',
                        'project_name':'sue',
                        'image_size':512,
                        'min_size':416,
                        'max_size':416,
                        'data_dir':'D:/Ctu_Project/DL/Ctu_Chainer_DL/dataset/DataSet_Tablet/DataJson',
                        'train_split':0.9,
                        'batch_size':2,
                        'model_name':'deeplab_resnet18',
                        'alpha':0.25,
                        'pre_model':'',
                        'encoding':'gbk',
                        'aux':False,
                        'n_processes':None,
                        'train_num':3,
                        'learning_rate':0.001,
                        'optimizer_type':'adam'

                    }
                    res_json = json.loads(requests.post(url='http://127.0.0.1:12345/start_train', data=json.dumps(postData),headers=header).text)
                    print("请求正常:",res_json)     
                    if res_json['return_value']=='1':
                        break
                except Exception as e:
                    print("请求异常:",str(e))
                
                time.sleep(1)

            # # 停止训练
            # while True:
            #     time.sleep(10)
            #     try:
            #         header = {"Content-Type": "application/json;charset=UTF-8"}
            #         postData = {
            #             'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
            #             'snid':'a76543dgfdsacc'
            #         }
            #         res_json = json.loads(requests.post(url='http://127.0.0.1:12345/stop_train', data=json.dumps(postData),headers=header).text)
            #         print("请求正常:",res_json)     
            #         if res_json['return_value']=='1':
            #             break
            #     except Exception as e:
            #         print("请求异常:",str(e))
                
            #     time.sleep(1)
            
            # 获取进度
            while True:
                try:
                    header = {"Content-Type": "application/json;charset=UTF-8"}
                    postData = {
                        'request_time':datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        'snid':'a76543dgfdsacc'
                    }
                    res_json = json.loads(requests.post(url='http://127.0.0.1:12345/get_progress', data=json.dumps(postData),headers=header).text)
                    print("请求正常:",res_json['message'])     
                except Exception as e:
                    print("请求异常:",str(e))
                time.sleep(1)
        while True:
            time.sleep(0.5)
    else:
        ctu = ctu_api_traintor(
            lic_file='license.lic',
            result_model = 'ctu_dl',
            server_host='0.0.0.0', 
            server_port=12345
        )
        ctu_model = ctu_api_predictor(
            use_gpu = '0',
            project_name = 'ctu_project',
            pool_dict = None,
            lic_file = 'license.lic',
            func_tool = get_return_func,
            server_port=54321
        )
        while True:
            time.sleep(0.01)