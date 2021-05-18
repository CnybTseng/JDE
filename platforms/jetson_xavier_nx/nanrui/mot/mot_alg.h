/**
 * @file mot_alg.h
 * @brief Multiple object tracking.
 * @author Zhiwei Zeng
 * @date 2021-5-14
 * @version v1.0
 * @copyright Copyright (c) 2004-2021 Chengdu Sihan Technology Co., Ltd
 */

#ifndef MOT_ALG_H_
#define MOT_ALG_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

int mot_alg_ability_init(void);
void mot_alg_ability_exit(void);
void *mot_alg_ability_data_fetch(void *input);
void *mot_alg_ability_inference(void *input);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // MOT_ALG_H_