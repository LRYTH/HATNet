from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from torch.utils.tensorboard import SummaryWriter

import time

import traceback
from collections import defaultdict

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
import skimage.io
import captioning.utils.eval_utils as eval_utils
import captioning.utils.misc as utils
from captioning.utils.rewards import init_scorer, get_self_critical_reward
from captioning.modules.loss_wrapper import *


# ===== 彩色输出工具 =====
def color_text(text, color=None, bg=None, style=None):
    color_map = {
        'red': 31, 'green': 32, 'yellow': 33,
        'blue': 34, 'magenta': 35, 'cyan': 36, 'white': 37
    }
    bg_map = {
        'red': 41, 'green': 42, 'yellow': 43,
        'blue': 44, 'magenta': 45, 'cyan': 46, 'white': 47
    }
    style_map = {
        'bold': 1, 'underline': 4
    }

    codes = []
    if style in style_map:
        codes.append(str(style_map[style]))
    if color in color_map:
        codes.append(str(color_map[color]))
    if bg in bg_map:
        codes.append(str(bg_map[bg]))

    return "\033[" + ";".join(codes) + "m" + str(text) + "\033[0m"


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)  #
    np.random.seed(seed)  #
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed(seed)  #
    torch.cuda.manual_seed_all(seed)  #


def train(opt):
    set_random_seed(42)  #

    # ===== GPU 初始化 =====
    if hasattr(opt, 'gpu_ids') and opt.gpu_ids != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids)
        gpu_ids = [int(x) for x in str(opt.gpu_ids).split(',')]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    use_multi_gpu = len(gpu_ids) > 1
    print(color_text(f"[GPU] 使用设备 ID: {gpu_ids} | {'多卡' if use_multi_gpu else '单卡'}模式", 'cyan', style='bold'))

    ################################
    # Build dataloader
    ################################
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # Initialize infos
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }
    # Load old infos(if there is) and check if models are compatible
    load_path = os.path.join(opt.start_from, 'infos_' + opt.id + "" + '.pkl')
    if opt.start_from is not None and os.path.isfile(load_path):
        with open(load_path, 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt,
                                                                    checkme), "Command line argument and saved model disagree on '%s' " % checkme
    infos['opt'] = opt

    #########################
    # Build logger
    #########################
    # naive dict logger
    histories = defaultdict(dict)
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))

    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    ##########################
    # Build model
    ##########################
    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()

    # print(model)

    del opt.vocab
    # Load pretrained weights:
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'model.pth')):
        path = os.path.join(opt.start_from, 'model.pth')
        model.load_state_dict(torch.load(path))

    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    lw_model = LossWrapper(model, opt)
    # Wrap with dataparallel

    # ----------------多卡运行------------------
    # dp_model = torch.nn.DataParallel(model)
    # dp_model.vocab = getattr(model, 'vocab', None)  # nasty
    # dp_lw_model = torch.nn.DataParallel(lw_model)

    # ----------------单卡卡运行------------------
    # dp_model = model
    # dp_model.vocab = getattr(model, 'vocab', None)
    # dp_lw_model = lw_model

    # ===== 单卡 / 多卡自动切换 =====
    if use_multi_gpu:
        dp_model = torch.nn.DataParallel(model, device_ids=list(range(len(gpu_ids))))
        dp_lw_model = torch.nn.DataParallel(lw_model, device_ids=list(range(len(gpu_ids))))
        print(color_text(f"[GPU] DataParallel 启动，device_ids={list(range(len(gpu_ids)))}", 'green', style='bold'))
    else:
        dp_model = model
        dp_lw_model = lw_model
        print(color_text("[GPU] 单卡运行", 'green', style='bold'))

    dp_model.vocab = getattr(model, 'vocab', None)

    ##########################
    #  Build optimizer
    ##########################

    if opt.noamopt:
        assert opt.caption_model in ['transformer', 'bert', 'm2transformer', 'georsclip',
                                     'qwen'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer,
                                            factor=opt.reduce_on_plateau_factor,
                                            patience=opt.reduce_on_plateau_patience)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    #########################
    # Get ready to start
    #########################
    iteration = infos['iter']
    epoch = infos['epoch']
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {
            split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]} for split in
            ['train', 'val', 'test']}
    loader.load_state_dict(infos['loader_state_dict'])
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    if opt.noamopt:
        optimizer._step = iteration
    # flag indicating finish of an epoch
    # Always set to True at the beginning to initialize the lr or etc.
    epoch_done = True
    # Assure in training mode
    dp_lw_model.train()

    # previous iteration validation loss
    prev_loss = 0
    loss_count = 0

    # Start training
    try:
        while True:
            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break

            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                # If start self critical training
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False

                # If start structure loss training
                if opt.structure_after != -1 and epoch >= opt.structure_after:
                    struc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    struc_flag = False
                if opt.drop_worst_after != -1 and epoch >= opt.drop_worst_after:
                    drop_worst_flag = True
                else:
                    drop_worst_flag = False

                epoch_done = False

            start = time.time()
            if opt.use_warmup and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * (iteration + 1) / opt.noamopt_warmup
                utils.set_lr(optimizer, opt.current_lr)
            data = loader.get_batch('train')
            print('Read data:', time.time() - start)
            torch.cuda.synchronize()
            start = time.time()

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            optimizer.zero_grad()
            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'],
                                    torch.arange(0, len(data['gts'])), sc_flag, struc_flag, drop_worst_flag)
            if not drop_worst_flag:
                loss = model_out['loss'].mean()
            else:
                loss = model_out['loss']
                loss = torch.topk(loss, k=int(loss.shape[0] * (1 - opt.drop_worst_rate)), largest=False)[0].mean()

            loss.backward()
            if opt.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' % (opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            end = time.time()
            if struc_flag:  # SL
                print(
                    "iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, model_out['lm_loss'].mean().item(),
                            model_out['struc_loss'].mean().item(), end - start))
            elif not sc_flag:  # CE
                # 原色输出
                # print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                #     .format(iteration, epoch, train_loss, end - start))
                # ===============彩色输出文本
                print(
                    f"{color_text('iter', 'cyan')} {color_text(iteration, 'yellow')} "
                    f"({color_text('epoch', 'cyan')} {color_text(epoch, 'magenta', style='bold')}), "
                    f"{color_text('train_loss', 'green')} = {color_text(f'{train_loss:.3f}', 'white', style='bold')}, "
                    f"{color_text('time/batch', 'blue')} = {color_text(f'{end - start:.3f}', 'yellow')}"
                )
            else:  # SC
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, model_out['reward'].mean(), end - start))

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                tb_summary_writer.add_scalar('scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    tb_summary_writer.add_scalar('avg_reward', model_out['reward'].mean(), iteration)
                elif struc_flag:
                    tb_summary_writer.add_scalar('lm_loss', model_out['lm_loss'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('struc_loss', model_out['struc_loss'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('reward', model_out['reward'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('reward_var', model_out['reward'].var(1).mean(), iteration)

                histories['loss_history'][iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                histories['lr_history'][iteration] = opt.current_lr
                histories['ss_prob_history'][iteration] = model.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()

            # make evaluation on validation set, and save model
            if (iteration % opt.save_checkpoint_every == 0 and not opt.save_every_epoch) or (
                    epoch_done and opt.save_every_epoch):
                # eval model
                eval_kwargs = {'split': 'val', 'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, lw_model.crit, loader,
                                                                          eval_kwargs)  # lw_model.crit 是CE
                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)
                # Write validation result into summary
                tb_summary_writer.add_scalar('validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k, v in lang_stats.items():
                        tb_summary_writer.add_scalar(k, v, iteration)
                histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats,
                                                              'predictions': predictions}

                # Save model if is improving on validation result

                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score

                utils.save_checkpoint(opt, model, infos, optimizer, histories)

                if opt.save_history_ckpt:
                    utils.save_checkpoint(opt, model, infos, optimizer,
                                          append=str(epoch) if opt.save_every_epoch else str(iteration))

                if best_flag:
                    utils.save_checkpoint(opt, model, infos, optimizer, append='best')

                # ===============不早停====================
                if val_loss > prev_loss:
                    loss_count += 1
                else:
                    loss_count = 0
                if loss_count >= 2:
                    # print("val_loss continuous raise {} times".format(loss_count))
                    # 带有红色提示的 验证损失持续增大
                    print("\033[31m" + "val_loss continuous raise {} times".format(loss_count) + "\033[0m")
                prev_loss = val_loss

                # ===============早停机制==================
                # # Early Stopping 逻辑
                # if val_loss > prev_loss:
                #     loss_count += 1
                # else:
                #     loss_count = 0
                #
                # if loss_count >= 2:
                #     print("\033[31m" + "val_loss continuous raise {} times".format(loss_count) + "\033[0m")
                #
                # # 🚨 连续3次上涨 → 提前停止
                # if loss_count >= 3:
                #     print("\033[41m\033[97m" + "Early stopping triggered! Stop training." + "\033[0m")
                #
                #     # 保存当前状态（保险）
                #     utils.save_checkpoint(opt, model, infos, optimizer, append='early_stop')
                #
                #     break  # 直接跳出训练循环
                #
                # prev_loss = val_loss


    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(opt, model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


opt = opts.parse_opt()

train(opt)
