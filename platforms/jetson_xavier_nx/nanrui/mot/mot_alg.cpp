/**
 * @file mot_alg.cpp
 * @brief Multiple object tracking.
 * @author Zhiwei Zeng
 * @date 2021-5-14
 * @version v1.0
 * @copyright Copyright (c) 2004-2021 Chengdu Sihan Technology Co., Ltd
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include <map>
#include <thread>
#include <vector>
#include "opencv2/opencv.hpp"

#include "algsdk.h"
#include "alg_osd.h"
#include "alg_data_hosting.h"
#include "alg_msg_gateway.h"
#include "alg_camera_control.h"
#include "mot_alg.h"
#include "mot.h"
#include "log.h"

static std::map<void *, mot::MOT_Result> motres;

static void *load_model(int dev_id)
{
    void *model_handle = nullptr;
    mot::load_mot_model("config.yaml", &model_handle);
    return model_handle;
}

static void unload_model(void *model_handle)
{
    mot::unload_mot_model(model_handle);
}

static void on_message(const char *topic, const char *from, const char *to, void *payload, int payloadlen)
{
    fprintf(stdout, "topic: %s, from: %s, to: %s\n", topic, from, to);
}

int mot_alg_ability_init(void)
{
    pub_info("mot_alg_ability_init ...\n");
    int i = 0;
    void *model_handle = nullptr;
    void **model_handle_addr = nullptr;
    
    // Set image format as BGR.
    algsdk_set_image_type(IMAGE_BGR);
    
    // Set frame frequency as 25Hz.
    algsdk_set_default_frequency(25);
    
    // Get usable processing units.
    ALGSDK_PROCESS_UNIT_LIST_ST *process_unit = algsdk_get_process_unit();
    if (process_unit && process_unit->unit_list) {
        for (i = 0; i < process_unit->unit_num; ++i) {
            pub_info("process unit %d: id=%d, percent=%d\n", i, process_unit->unit_list[i].id,
                process_unit->unit_list[i].percent);
        }
        model_handle = load_model(process_unit->unit_list[0].id);
        pub_info("load_model %p\n", model_handle);
        // Allocate public parameter space for storing model handle.
        model_handle_addr = (void **)algsdk_get_public_param((char *)"model_handle", SHARE_IN_TASK, sizeof(model_handle));
        if (model_handle_addr) {
            *model_handle_addr = model_handle;
        } else {
            pub_error("algsdk_get_public_param(model_handle) failed\n");
            return -1;
        }
    }
    
    alg_msg_gateway_subscribe("topic", nullptr, nullptr);
    alg_msg_gateway_callback_set(on_message);
    pub_info("mot_alg_ability_init done\n");
 
    return 0;
}

void mot_alg_ability_exit(void)
{
    pub_info("mot_alg_ability_exit ...\n");
    void *model_handle = nullptr;
    void **model_handle_addr = nullptr;
    
    model_handle_addr = (void **)algsdk_get_public_param((char *)"model_handle", SHARE_IN_TASK, sizeof(model_handle));
    if(model_handle_addr) {
        model_handle = *model_handle_addr;
    } else {
        pub_error("algsdk_get_public_param(model_handle) failed\n");
        return;
    }

    pub_info("unload_model %p\n", model_handle);
    unload_model(model_handle);
    pub_info("mot_alg_ability_exit done\n");
}

void *mot_alg_ability_data_fetch(void *input)
{
    ALGSDK_SOURCE_HANDLE_ST *source_handle = algsdk_get_source_handle();
    if (!source_handle) { 
        pub_error("algsdk_get_source_handle failed\n");
        return nullptr;
    }
    
    ALGSDK_IMAGE_ST *source_image = algsdk_get_source_image(source_handle->handle_group[0]);
    if (!source_image) {
        pub_error("algsdk_get_source_image failed\n");
        return nullptr;
    }
    
    ALGSDK_RULE_LIST_ST *rule = algsdk_get_source_rule(source_handle->handle_group[0]);
    
    char *url = algsdk_get_source_url(source_handle->handle_group[0]);
    if (!url) {
        pub_error("algsdk_get_source_url failed\n");
        return nullptr;
    }
    
    void **output = (void **)algsdk_malloc(sizeof(source_image) + sizeof(rule) + sizeof(char *));
    if (!output) {
        pub_error("algsdk_malloc failed\n");
        return nullptr;
    }

    output[0] = source_image;
    output[1] = rule;
    output[2] = url;
    
    return output;
}

void *mot_alg_ability_inference(void *input)
{    
    // Parse image, rule, and url.
    void **real_input = (void **)input;
    ALGSDK_IMAGE_ST *source_image = (ALGSDK_IMAGE_ST *)real_input[0];
    ALGSDK_RULE_LIST_ST *rule = (ALGSDK_RULE_LIST_ST *)real_input[1];
    char *source_url = (char *)real_input[2];

    ALG_DATA_HOSTING_T *alarm_image = nullptr;
    ALGSDK_POINT_ST point = {0};
    ALGSDK_OBJECT_BOX_ST obj_box = {0};
    ALGSDK_OBJ_LIST_ST obj_list = {0};
    ALG_ALARM_EVENT_PUSH_ST alarm_push = {0};
    ALG_ALARM_IMAGE_DESC_ST alarm_push_image = {0};

    int ret = 0;
    void *model_handle = nullptr;
    void **model_handle_addr = nullptr;
    
    model_handle_addr = (void **)algsdk_get_public_param((char *)"model_handle", SHARE_IN_TASK, sizeof(model_handle));
    if (model_handle_addr) {
        model_handle = *model_handle_addr;
    } else {
        pub_error("algsdk_get_public_param(model_handle) failed\n");
        return nullptr;
    }
    
    // Neural network inference.
    pub_debug("forward_mot_model %p\n", model_handle);
    mot::forward_mot_model((unsigned char *)source_image->image_data.data[0], source_image->width,
        source_image->height, source_image->image_data.step[0], motres[model_handle], model_handle); 
    
    // Push metadata.    
    std::vector<mot::MOT_Track>::iterator riter;
    for (riter = motres[model_handle].begin(); riter != motres[model_handle].end(); riter++) {
        std::deque<mot::MOT_Rect>::iterator iter;
        for (iter = riter->rects.begin(); iter != riter->rects.end(); iter++) {
            int l = static_cast<int>(iter->left);
            int t = static_cast<int>(iter->top);
            int r = static_cast<int>(iter->right);
            int b = static_cast<int>(iter->bottom);
            float h = iter->bottom > iter->top ? iter->bottom - iter->top : 1e-5f;
            float ar = (iter->right - iter->left) / h;   // aspect ratio
            if ((l == 0 && t == 0 && r == 0 && b == 0) || ar > 1.f)
                break;
            
            // Points in clockwise order.
            algsdk_set_point(&point, 0, l, t, nullptr);     // top left
            algsdk_add_point_to_obj_box(&obj_box, &point);
            algsdk_set_point(&point, 1, r, t, nullptr);     // top right
            algsdk_add_point_to_obj_box(&obj_box, &point);
            algsdk_set_point(&point, 2, r, b, nullptr);     // bottom right
            algsdk_add_point_to_obj_box(&obj_box, &point);
            algsdk_set_point(&point, 3, l, b, nullptr);     // bottom left
            algsdk_add_point_to_obj_box(&obj_box, &point);
            
            char desc[128] = {0};
            sprintf(desc, "%d", riter->identifier);
            algsdk_set_obj_box(&obj_box, desc, OBJ_COLOR_YELLOW);
            algsdk_add_obj_box_to_list(&obj_list, &obj_box);
            break;  // only draw the latest bounding box
        }
    }
    
    algsdk_push_metadata((char *)"mot metadata", &obj_list, source_url, source_image->pts);
    
    // Alarm handle.
    if (algsdk_alarm_time_check() == 0) {
        // Overlap OSD on alarm image.
        char name[128] = {0};
        sprintf(name, "%lld.jpg", source_image->pts);
        alarm_image = alg_osd_get_osd_image(name, source_image, &obj_list);
        if (!alarm_image) {
            pub_error("alg_osd_get_osd_image failed\n");
            goto EXIT_1;
        }
        
        // Push alarm event.
        snprintf(alarm_push.desc, sizeof(alarm_push.desc), "%s", "mot alarm");
        alarm_push.image_num = 1;
        alarm_push.image_list = &alarm_push_image;        
        snprintf(alarm_push_image.name, sizeof(alarm_push_image.name), "%s", name);
        alarm_push_image.width = source_image->width;
        alarm_push_image.height = source_image->height;
        alarm_push_image.size = alarm_image->length;

        ret = algsdk_alarm_event_push(&alarm_push);
        if (1 == ret) {
            pub_error("algsdk_alarm_event_push failed\n");
            goto EXIT_1;
        }
        
        // Alarm image hosting.
        pub_info("alarm_image type = %d, title = %s, length = %lu\n",
            alarm_image->type, alarm_image->title, alarm_image->length);
        ret = alg_data_hosting_put(alarm_image);
    }

EXIT_1:   
    // Release image, rule, and url.
    algsdk_release_source_image(source_image);
    alg_osd_release_osd_image(alarm_image);
    algsdk_release_source_rule(rule);
    if (source_url) {
        free(source_url);
        source_url = nullptr;
    }
    algsdk_free_obj_list(&obj_list);
    
    return nullptr;
}