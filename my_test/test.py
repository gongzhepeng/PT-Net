import numpy as np
import torch
import os
import sys
import cv2
from loss import MatchLoss
from tqdm import tqdm
from dataset import collate_fn, CorrespondencesDataset   # 没啥用
from utils import compute_pose_error, pose_auc, estimate_pose_norm_kpts, estimate_pose_from_E, get_pool_result, tocuda, pose_metric, eval_nondecompose, eval_decompose
from config import get_config

sys.path.append('../core')
from convmatch import ConvMatch

#torch.set_grad_enabled(False)
#torch.manual_seed(0)

def inlier_test(config, polar_dis, inlier_mask):
    polar_dis = polar_dis.reshape(inlier_mask.shape).unsqueeze(0)
    inlier_mask = torch.from_numpy(inlier_mask).type(torch.float32)
    is_pos = (polar_dis < config.obj_geod_th).type(inlier_mask.type())
    is_neg = (polar_dis >= config.obj_geod_th).type(inlier_mask.type())
    precision = torch.mean(
        torch.sum(inlier_mask * is_pos, dim=1) /
        (torch.sum(inlier_mask * (is_pos + is_neg), dim=1)+1e-15)
    )
    recall = torch.mean(
        torch.sum(inlier_mask * is_pos, dim=1) /
        torch.sum(is_pos, dim=1)
    )
    f_scores = 2*precision*recall/(precision+recall+1e-15)

    return precision, recall, f_scores

def test_sample(args):
    _xs, _dR, _dt, _e_hat, _y_hat, _y_gt, config, = args
    _xs = _xs.reshape(-1, 4).astype('float64')
    _dR, _dt = _dR.astype('float64').reshape(3,3), _dt.astype('float64')
    _y_hat_out = _y_hat.flatten().astype('float64')
    e_hat_out = _e_hat.flatten().astype('float64')

    _x1 = _xs[:, :2]
    _x2 = _xs[:, 2:]
    # current validity from network
    _valid = _y_hat_out
    # choose top ones (get validity threshold)
    _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
    _mask_before = _valid >= max(0, _valid_th)

    if not config.use_ransac:
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_nondecompose(_x1, _x2, e_hat_out, _dR, _dt, _y_hat_out)
    else:
        # actually not use prob here since probs is None
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_decompose(_x1, _x2, _dR, _dt, mask=_mask_before, method=cv2.RANSAC, \
            probs=None, weighted=False, use_prob=True)
    if _R_hat is None:
        _R_hat = np.random.randn(3,3)
        _t_hat = np.random.randn(3,1)
    return [float(_err_q), float(_err_t), float(_num_inlier), _R_hat.reshape(1,-1), _t_hat.reshape(1,-1)]

def denorm(x, T):
    x = (x - np.array([T[0,2], T[1,2]])) / np.asarray([T[0,0], T[1,1]])
    return x

def test(config):

    model = ConvMatch(config)

    test_dataset = CorrespondencesDataset(config.data_te, config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    model = ConvMatch(config)

    save_file_best = os.path.join(config.model_file, "model_best.pth")    # 把model_best的文件路径载入
    if not os.path.exists(save_file_best):
        print("Model File {} does not exist! Quiting".format(save_file_best))
        exit(1)
    # Restore model  重建模型
    checkpoint = torch.load(save_file_best)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    cur_global_step = 0
    model.eval()
    print('generated parameters:', sum(param.numel() for param in model.parameters()))
    match_loss = MatchLoss(config)
    loader_iter = iter(test_loader)
    network_infor_list = ["geo_losses", "cla_losses", "l2_losses", 'precisions', 'recalls', 'f_scores']
    network_info = {info: [] for info in network_infor_list}
    results, pool_arg = [], []
    eval_step, eval_step_i, num_processor = 100, 0, 8
    with torch.no_grad():
        for test_data in loader_iter:
            test_data = tocuda(test_data)
            res_logits, res_e_hat = model(test_data)
            y_hat, e_hat = res_logits[-1], res_e_hat[-1]
            loss, geo_loss, cla_loss, l2_loss, prec, rec = match_loss.run(cur_global_step, test_data, y_hat, e_hat)
            info = [geo_loss, cla_loss, l2_loss, prec, rec, 2 * prec * rec / (prec + rec + 1e-15)]
            for info_idx, value in enumerate(info):
                network_info[network_infor_list[info_idx]].append(value)

            if config.use_fundamental:
                # unnorm F
                e_hat = torch.matmul(torch.matmul(test_data['T2s'].transpose(1,2), e_hat.reshape(-1,3,3)),test_data['T1s'])
                # get essential matrix from fundamental matrix
                e_hat = torch.matmul(torch.matmul(test_data['K2s'].transpose(1,2), e_hat.reshape(-1,3,3)),test_data['K1s']).reshape(-1,9)
                e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)

            # pose_err
            for batch_idx in range(e_hat.shape[0]):
                test_xs = test_data['xs'][batch_idx].detach().cpu().numpy()
                if config.use_fundamental: # back to original
                    x1, x2 = test_xs[0,:,:2], test_xs[0,:,2:4]
                    T1, T2 = test_data['T1s'][batch_idx].cpu().numpy(), test_data['T2s'][batch_idx].cpu().numpy()
                    x1, x2 = denorm(x1, T1), denorm(x2, T2) # denormalize coordinate
                    K1, K2 = test_data['K1s'][batch_idx].cpu().numpy(), test_data['K2s'][batch_idx].cpu().numpy()
                    x1, x2 = denorm(x1, K1), denorm(x2, K2) # normalize coordiante with intrinsic
                    test_xs = np.concatenate([x1,x2],axis=-1).reshape(1,-1,4)

                pool_arg += [(test_xs, test_data['Rs'][batch_idx].detach().cpu().numpy(), \
                              test_data['ts'][batch_idx].detach().cpu().numpy(),
                              e_hat[batch_idx].detach().cpu().numpy(), \
                              y_hat[batch_idx].detach().cpu().numpy(), \
                              test_data['ys'][batch_idx, :, 0].detach().cpu().numpy(), config)]

                eval_step_i += 1
                if eval_step_i % eval_step == 0:
                    results += get_pool_result(num_processor, test_sample, pool_arg)
                    pool_arg = []
            # pose_err

        if len(pool_arg) > 0:
                results += get_pool_result(num_processor, test_sample, pool_arg)

    measure_list = ["err_q", "err_t", "num", 'R_hat', 't_hat']
    eval_res = {}
    for measure_idx, measure in enumerate(measure_list):
        eval_res[measure] = np.asarray([result[measure_idx] for result in results])

    ret_val = pose_metric(eval_res)
    precision = torch.mean(torch.Tensor(network_info['precisions']))
    recall = torch.mean(torch.Tensor(network_info['recalls']))
    f_scores = 2*precision*recall/(precision+recall+1e-15)
    #f_scores = torch.mean(torch.Tensor(network_info['f_scores']))

    #print('Evaluation Results (mean over {} pairs):'.format(len(test_loader)))
    print('MAP@5\t MAP@10\t MAP@20\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(ret_val[0]*100, ret_val[1]*100, ret_val[3]*100))
    print('Prec\t Rec\t F1\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(precision*100, recall*100, f_scores*100))
    return

if __name__ == '__main__':
    config, unparsed = get_config()
    test(config)
