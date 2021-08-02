# #본 코드는 그냥 main 에 붙여 넣기만 하면됨.
##########################################



### untargeted_FGSM 만들기
if exists(f'./{ATTACK_load_path}/{ATTACK_EPS}_test') and exists(f'./{ATTACK_load_path}/{ATTACK_EPS}_label'):
    attack_test = pickle.load(open(f'./{ATTACK_load_path}/{ATTACK_EPS}_test','rb'))
    attack_label = pickle.load(open(f'./{ATTACK_load_path}/{ATTACK_EPS}_label','rb'))

else:
    attack_test, attack_label = [], []

    for i in trange(len(x_test)):
        
        adv_data = eval('untargeted_fgsm')(mnist_model, x_test[i], ATTACK_EPS) # (28, 28, 1)
        attack_test.append(adv_data)

        pred_adv_data = mnist_model.predict(tf.expand_dims(adv_data, 0))
        pred_adv_data = np.argmax(pred_adv_data)

        if y_test[i] != pred_adv_data:
            attack_label.append(1)
        else:
            attack_label.append(0)

    attack_test, attack_label = np.array(attack_test), np.array(attack_label)

    pickle.dump(attack_test, open(f'./{ATTACK_load_path}/{ATTACK_EPS}_test','wb'))
    pickle.dump(attack_label, open(f'./{ATTACK_load_path}/{ATTACK_EPS}_label','wb'))




# ### untargeted_PGD 만들기
# if exists(f'./{ATTACK_load_path}/{ATTACK_EPS}_test') and exists(f'./{ATTACK_load_path}/{ATTACK_EPS}_label'):
#     attack_test = pickle.load(open(f'./{ATTACK_load_path}/{ATTACK_EPS}_test','rb'))
#     attack_label = pickle.load(open(f'./{ATTACK_load_path}/{ATTACK_EPS}_label','rb'))

# else:
#     attack_test, attack_label = [], []

#     for i in trange(len(x_test)):
        
#         adv_data = eval('untargeted_pgd')(mnist_model, x_test[i], ATTACK_EPS) # (28, 28, 1)
#         attack_test.append(adv_data)

#         pred_adv_data = mnist_model.predict(tf.expand_dims(adv_data, 0))
#         pred_adv_data = np.argmax(pred_adv_data)

#         if y_test[i] != pred_adv_data:
#             attack_label.append(1)
#         else:
#             attack_label.append(0)

#     attack_test, attack_label = np.array(attack_test), np.array(attack_label)

#     pickle.dump(attack_test, open(f'./{ATTACK_load_path}/{ATTACK_EPS}_test','wb'))
#     pickle.dump(attack_label, open(f'./{ATTACK_load_path}/{ATTACK_EPS}_label','wb'))





# ### untargeted_MIM 만들기
# if exists(f'./{ATTACK_load_path}/{ATTACK_EPS}_test') and exists(f'./{ATTACK_load_path}/{ATTACK_EPS}_label'):
#     attack_test = pickle.load(open(f'./{ATTACK_load_path}/{ATTACK_EPS}_test','rb'))
#     attack_label = pickle.load(open(f'./{ATTACK_load_path}/{ATTACK_EPS}_label','rb'))

# else:
#     attack_test, attack_label = [], []

#     for i in trange(len(x_test)):
        
#         adv_data = eval('untargeted_mim')(mnist_model, x_test[i], ATTACK_EPS) # (28, 28, 1)
#         attack_test.append(adv_data)

#         pred_adv_data = mnist_model.predict(tf.expand_dims(adv_data, 0))
#         pred_adv_data = np.argmax(pred_adv_data)

#         if y_test[i] != pred_adv_data:
#             attack_label.append(1)
#         else:
#             attack_label.append(0)

#     attack_test, attack_label = np.array(attack_test), np.array(attack_label)

#     pickle.dump(attack_test, open(f'./{ATTACK_load_path}/{ATTACK_EPS}_test','wb'))
#     pickle.dump(attack_label, open(f'./{ATTACK_load_path}/{ATTACK_EPS}_label','wb'))




# ### untargeted_CW 만들기
# if exists(f'./{ATTACK_load_path}/test') and exists(f'./{ATTACK_load_path}/label'):
#     attack_test = pickle.load(open(f'./{ATTACK_load_path}/test','rb'))
#     attack_label = pickle.load(open(f'./{ATTACK_load_path}/label','rb'))

# else:
#     attack_test, attack_label = [], []

#     for i in trange(len(x_test)):
        
#         adv_data = eval('untargeted_cw')(mnist_model, x_test[i]) # (28, 28, 1)
#         attack_test.append(adv_data)

#         pred_adv_data = mnist_model.predict(tf.expand_dims(adv_data, 0))
#         pred_adv_data = np.argmax(pred_adv_data)

#         if y_test[i] != pred_adv_data:
#             attack_label.append(1)
#         else:
#             attack_label.append(0)

#     attack_test, attack_label = np.array(attack_test), np.array(attack_label)

#     pickle.dump(attack_test, open(f'./{ATTACK_load_path}/test','wb'))
#     pickle.dump(attack_label, open(f'./{ATTACK_load_path}/label','wb'))



### Saliency 만들기

# if exists(f'./dataset/normal_saliency/normal_train'):
#     g_train = pickle.load(open(f'./dataset/normal_saliency/train','rb'))

# else:
#     g_train= []

#     for i in trange(len(x_train)):
#         g_train.append(eval('vanilla_saliency')(mnist_model, x_train[i])) # (28, 28, 1)

#     g_train = np.array(g_train)

#     pickle.dump(g_train, open(f'./dataset/normal_saliency/train','wb'))



# ### IG 만들기

# if exists(f'./dataset/normal_ig/normal_train'):
#     ig_train = pickle.load(open(f'./dataset/normal_ig/normal_train','rb'))

# else:
#     print("IG 시작")
#     ig_train, ig_test = [], []

#     for i in trange(len(x_train)):
#         ig_train.append(eval('ig')(mnist_model, x_train[i])) # (28, 28, 1)

#     ig_train = np.array(ig_train)

#     pickle.dump(ig_train, open(f'./dataset/normal_ig/normal_train','wb'))



# ## smooth_Saliency 만들기

# if exists(f'./dataset/normal_smooth_saliency/normal_smooth_train'):
#     smooth_g_train = pickle.load(open(f'./dataset/normal_smooth_saliency/normal_smooth_train','rb'))

# else:
#     smooth_g_train= []

#     for i in trange(len(x_train)):
#         smooth_g_train.append(eval('smooth_saliency')(mnist_model, x_train[i])) # (28, 28, 1)

#     smooth_g_train = np.array(smooth_g_train)

#     pickle.dump(smooth_g_train, open(f'./dataset/normal_smooth_saliency/normal_smooth_train','wb'))





# ## smooth_IG 만들기

# if exists(f'./dataset/normal_smooth_ig/normal_smooth_train'):
#     smooth_ig_train = pickle.load(open(f'./dataset/normal_smooth_ig/normal_smooth_train','rb'))

# else:
#     smooth_ig_train= []

#     for i in trange(len(x_train)):
#         smooth_ig_train.append(eval('smooth_ig')(mnist_model, x_train[i])) # (28, 28, 1)

#     smooth_ig_train = np.array(smooth_ig_train)

#     pickle.dump(smooth_ig_train, open(f'./dataset/normal_smooth_ig/normal_smooth_train','wb'))

