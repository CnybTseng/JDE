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