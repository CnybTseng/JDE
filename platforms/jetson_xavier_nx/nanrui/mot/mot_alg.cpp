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

static std::map<std::thread::id, mot::MOT_Result> motres;

static bool flag = false;

int load_model(int dev_id)
{
    fprintf(stdout, "load_model\r\n");
    // mot::load_mot_model("config.yaml");
    return 0;
}

void unload_model(int model_handle)
{
    mot::unload_mot_model();
}

void on_message(const char *topic, const char *from, const char *to, void *payload, int payloadlen)
{
    fprintf(stderr, "topic: %s, from: %s, to: %s\n", topic, from, to);
}

int mot_alg_ability_init(void)
{
    fprintf(stderr, "mot_alg_ability_init\r\n");
    int i = 0;
    int model_handle = 0;
    int *model_handle_addr = NULL;
    struct alg_data_hosting_t data;
    
    // Set image format as BGR.
    algsdk_set_image_type(IMAGE_BGR);
    
    // Set frame frequency as 25Hz.
    algsdk_set_default_frequency(25);
    
    // Get usable processing units.
    ALGSDK_PROCESS_UNIT_LIST_ST *process_unit = algsdk_get_process_unit();
    if (process_unit && process_unit->unit_list) {
        // Print all processing unit.
        for (i = 0; i < process_unit->unit_num; ++i) {
            printf("%d id=%d percent=%d\r\n", i,
                process_unit->unit_list[i].id,
                process_unit->unit_list[i].percent);
        }
        model_handle = load_model(process_unit->unit_list[0].id);
        // Allocate public parameter space for storing model handle.
        model_handle_addr = (int *)algsdk_get_public_param((char *)"model_handle", SHARE_IN_TASK, sizeof(model_handle));
        if (model_handle_addr)
            *model_handle_addr = model_handle;
    }
    
    // Structure data.
    data.type = 0;
    data.title = NULL;
    data.content = (void *)"[{\"Key\":\"nu\",\"Value\":1},{\"Key\":\"start\",\"Value\":{\"1\":\"1\",\"2\":{\"3\":2}}}]";
    data.length = strlen("[{\"Key\":\"nu\",\"Value\":1},{\"Key\":\"start\",\"Value\":{\"1\":\"1\",\"2\":{\"3\":2}}}]");
    
    alg_data_hosting_put(&data);
    
    alg_msg_gateway_subscribe("topic", NULL, NULL);
    alg_msg_gateway_callback_set(on_message);
    
    return 0;
}

void mot_alg_ability_exit(void)
{
    int model_handle = 0;
    int *model_handle_addr = NULL;
    char content[1024];
    struct alg_data_hosting_t data;

    alg_data_hosting_get_value("nu", content, sizeof(content));
    fprintf(stderr, "demo_alg_ability_exit: %s\r\n", content);
    alg_data_hosting_get_value("start", content, sizeof(content));
    fprintf(stderr, "demo_alg_ability_exit: %s\r\n", content);

    data.type = 0;
    data.title = NULL;
    data.content = (void *)"[{\"Key\":\"nu\",\"Value\":2}]";
    data.length = strlen("[{\"Key\":\"nu\",\"Value\":2}]");
    int ret = alg_data_hosting_put(&data);
    fprintf(stderr, "demo_alg_ability_exit:%d\r\n", ret);

    sleep(5);
    alg_data_hosting_get_value("nu", content, sizeof(content));
    fprintf(stderr, "demo_alg_ability_exit: %s\r\n", content);
    alg_data_hosting_get_value("start", content, sizeof(content));
    fprintf(stderr, "demo_alg_ability_exit: %s\r\n", content);
    
    model_handle_addr = (int *)algsdk_get_public_param((char *)"model_handle", SHARE_IN_TASK, sizeof(model_handle));
    if(model_handle_addr)
    {
        model_handle = *model_handle_addr;
    }
    
    unload_model(model_handle);
}

void *mot_alg_ability_data_fetch(void *input)
{
    ALGSDK_SOURCE_HANDLE_ST *source_handle = algsdk_get_source_handle();
    if (!source_handle) { 
        fprintf(stderr, "algsdk_get_source_handle failed\r\n");
        return NULL;
    }
    
    ALGSDK_IMAGE_ST *source_image = algsdk_get_source_image(source_handle->handle_group[0]);
    if (!source_image) {
        fprintf(stderr, "algsdk_get_source_image failed\r\n");
        return NULL;
    }
    
    ALGSDK_RULE_LIST_ST *rule = algsdk_get_source_rule(source_handle->handle_group[0]);
    
    char *url = algsdk_get_source_url(source_handle->handle_group[0]);
    if (!url) {
        fprintf(stderr, "algsdk_get_source_url failed\r\n");
        return NULL;
    }
    
    void **output = (void **)algsdk_malloc(sizeof(source_image) + sizeof(rule) + sizeof(char *));
    if (!output) {
        fprintf(stderr, "algsdk_malloc failed\r\n");
        return NULL;
    }

    output[0] = source_image;
    output[1] = rule;
    output[2] = url;
    
    return output;
}

void *mot_alg_ability_inference(void *input)
{
    if (flag == false) {
        mot::load_mot_model("config.yaml");
        flag = true;
    }
    
    // Parse image, rule, and url.
    void **real_input = (void **)input;
    ALGSDK_IMAGE_ST *source_image = (ALGSDK_IMAGE_ST *)real_input[0];
    ALGSDK_RULE_LIST_ST *rule = (ALGSDK_RULE_LIST_ST *)real_input[1];
    char *source_url = (char *)real_input[2];
    
    ALG_DATA_HOSTING_T *alarm_image = NULL;
    ALGSDK_POINT_ST point = {0};
    ALGSDK_OBJECT_BOX_ST obj_box = {0};
    ALGSDK_OBJ_LIST_ST obj_list = {0};
    ALG_ALARM_EVENT_PUSH_ST alarm_push = {0};
    ALG_ALARM_IMAGE_DESC_ST alarm_push_image = {0};

    int ret = 0;
    int model_handle = 0;
    int *model_handle_addr = NULL;
    // char *camera_handle = NULL;
   
    // Get public parameters
    char *history = (char *)algsdk_get_public_param((char *)"history", SHARE_IN_ATOM, 100);
    if (!history) {
        fprintf(stderr, "algsdk_get_public_param failed\r\n");
        return NULL;
    }
    
    fprintf(stdout, "history=%s\r\n", history);
    sprintf(history, "history time = %ld", time(NULL));
    
    model_handle_addr = (int *)algsdk_get_public_param((char *)"model_handle", SHARE_IN_TASK, sizeof(model_handle));
    if (model_handle_addr)
        model_handle = *model_handle_addr;
    
    // Neural network inference.
    std::thread::id tid = std::this_thread::get_id();
    fprintf(stdout, "width %d, height %d, stride %d\r\n", source_image->width, source_image->height, source_image->image_data.step[0]);
    mot::forward_mot_model((unsigned char *)source_image->image_data.data[0], source_image->width, source_image->height, source_image->image_data.step[0], motres[tid]); 
    fprintf(stdout, "forward_mot_model done\r\n");
    
    // Push metadata.    
    std::vector<mot::MOT_Track>::iterator riter;
    for (riter = motres[tid].begin(); riter != motres[tid].end(); riter++) {
        std::deque<mot::MOT_Rect>::iterator iter;
        for (iter = riter->rects.begin(); iter != riter->rects.end(); iter++) {
            int l = static_cast<int>(iter->left);
            int t = static_cast<int>(iter->top);
            int r = static_cast<int>(iter->right);
            int b = static_cast<int>(iter->bottom);
            if (l == 0 && t == 0 && r == 0 && b == 0)
                break;
            
            algsdk_set_point(&point, 0, l, t, NULL);
            algsdk_add_point_to_obj_box(&obj_box, &point);
            algsdk_set_point(&point, 1, r, t, NULL);
            algsdk_add_point_to_obj_box(&obj_box, &point);
            algsdk_set_point(&point, 2, r, b, NULL);
            algsdk_add_point_to_obj_box(&obj_box, &point);
            algsdk_set_point(&point, 3, l, b, NULL);
            algsdk_add_point_to_obj_box(&obj_box, &point);
            
            char desc[128] = {0};
            sprintf(desc, "%d", riter->identifier);
            algsdk_set_obj_box(&obj_box, desc, OBJ_COLOR_YELLOW);
            algsdk_add_obj_box_to_list(&obj_list, &obj_box);
            break;  // only draw the latest bounding box
        }
    }
    
    algsdk_push_metadata((char *)"mot metadata", &obj_list, source_url, source_image->pts);
    
    // Publish task event.
    if (algsdk_alarm_time_check() == 0) {
        // Overlap OSD on image.
        char name[128] = {0};
        sprintf(name, "%lld.jpg", source_image->pts);
        alarm_image = alg_osd_get_osd_image(name, source_image, &obj_list);
        if (!alarm_image) {
            fprintf(stderr, "alg_osd_get_osd_image failed\r\n");
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
            fprintf(stderr, "algsdk_alarm_event_push failed\r\n");
            goto EXIT_1;
        }
        
        // Alarm image trusteeship.
        fprintf(stdout, "before alg_data_hosting_put, alarm_image type = %d, title = %s, length = %lu\r\n",
            alarm_image->type, alarm_image->title, alarm_image->length);
        ret = alg_data_hosting_put(alarm_image);
        fprintf(stdout, "alg_data_hosting_put ret %d\r\n", ret);
    }
    
    alg_msg_gateway_publish((char *)"topic", history, NULL, history, strlen(history) + 1);
    
    // Camera control.
    // camera_handle = alg_camera_control_get_help(source_url);
    // if (camera_handle) {
    //     ret = alg_camera_control_handler(camera_handle, 0, time(NULL) % 10, 1, 1);
    //     fprintf(stdout, "alg_camera_control_handler return %d\r\n", ret);
    //     alg_camera_control_release_handle(camera_handle);
    //     camera_handle = NULL;
    // }

EXIT_1:   
    // Release memories for image, rule, and url.
    algsdk_release_source_image(source_image);
    alg_osd_release_osd_image(alarm_image);
    algsdk_release_source_rule(rule);
    if (source_url) {
        free(source_url);
        source_url = NULL;
    }
    algsdk_free_obj_list(&obj_list);
    
    return NULL;
}