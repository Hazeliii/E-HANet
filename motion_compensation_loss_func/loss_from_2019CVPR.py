import torch
import numpy as np
'''
实现2019_cvpr中的损失函数
input:
'''


def generate_ts_image(events, pre_flow, pre_valid):
    '''
    @param events: <class 'numpy.ndarry'> a set of events
    @param pre_flow: the prediction flow, torch.size([256, 256， 2])
    @param pre_valid: the mask of prediction flow，torch.size([256, 256])
    @return: images of the average timestamp at each pixel for each polarity  each img:torch.Size([1, 256, 256])
    '''
    height = 256
    width = 256
    events_num = events.shape[0]
    start_time = events[0][0]
    positive_accumulated_img = torch.zeros((height, width), dtype=torch.float).cuda()
    negative_accumulated_img = torch.zeros((height, width), dtype=torch.float).cuda()
    pos_per_pixel_num = torch.zeros((height, width), dtype=torch.float).cuda()
    neg_per_pixel_num = torch.zeros((height, width), dtype=torch.float).cuda()

    # normalizetion makes no difference
    for i in range(events_num):
        the_event = events[i]
        x = the_event[1] - 45.
        y = the_event[2] - 2.
        if (x in range(0, width)) and (y in range(0, height)) and pre_valid[int(y), int(x)]:

            p = the_event[3]  #{-1, 1}
            t = the_event[0]
            u = pre_flow[int(y), int(x)][0] / 50
            v = pre_flow[int(y), int(x)][1] / 50
            delta_T = float(t - start_time) / 1000  # s->ms
            x_ref = x - u * delta_T
            y_ref = y - v * delta_T
            x_ref_int = int(x_ref)
            y_ref_int = int(y_ref)  # 向下取整
            if x_ref_int in range(0, width-1) and y_ref_int in range(0, height-1):
                dx = x_ref - x_ref_int
                dy = y_ref - y_ref_int
                if p == 1:
                    positive_accumulated_img[y_ref_int, x_ref_int] += (1.-dx) * dy* t
                    positive_accumulated_img[y_ref_int, x_ref_int+1] += dx*(1-dy) * t
                    positive_accumulated_img[y_ref_int+1, x_ref_int] += (1. - dx) * dy * t
                    positive_accumulated_img[y_ref_int+1, x_ref_int + 1] += dx * dy * t

                    pos_per_pixel_num[y_ref_int, x_ref_int] += (1. - dx) * dy
                    pos_per_pixel_num[y_ref_int, x_ref_int + 1] += dx * (1 - dy)
                    pos_per_pixel_num[y_ref_int + 1, x_ref_int] += (1. - dx) * dy
                    pos_per_pixel_num[y_ref_int + 1, x_ref_int + 1] += dx * dy

                elif p == -1:
                    negative_accumulated_img[y_ref_int, x_ref_int] += (1. - dx) * dy * t
                    negative_accumulated_img[y_ref_int, x_ref_int + 1] += dx * (1 - dy) * t
                    negative_accumulated_img[y_ref_int + 1, x_ref_int] += (1. - dx) * dy * t
                    negative_accumulated_img[y_ref_int + 1, x_ref_int + 1] += dx * dy * t

                    neg_per_pixel_num[y_ref_int, x_ref_int] += (1. - dx) * dy
                    neg_per_pixel_num[y_ref_int, x_ref_int + 1] += dx * (1 - dy)
                    neg_per_pixel_num[y_ref_int + 1, x_ref_int] += (1. - dx) * dy
                    neg_per_pixel_num[y_ref_int + 1, x_ref_int + 1] += dx * dy
                else:
                    print('The polarity should be -1 or 1!!')
    pos_mask = (positive_accumulated_img != 0)
    pos_img_masked = torch.masked_select(positive_accumulated_img, pos_mask)
    pos_num_masked = torch.masked_select(pos_per_pixel_num, pos_mask)

    neg_mask = (negative_accumulated_img != 0)
    neg_img_masked = torch.masked_select(negative_accumulated_img, neg_mask)
    neg_num_masked = torch.masked_select(neg_per_pixel_num, neg_mask)

    ave_ts_pos_img = torch.div(pos_img_masked, pos_num_masked)
    ave_ts_neg_img = torch.div(neg_img_masked, neg_num_masked)

    return torch.square(ave_ts_pos_img), torch.square(ave_ts_neg_img)






