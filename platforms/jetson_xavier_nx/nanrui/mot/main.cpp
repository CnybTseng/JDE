#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "algsdk.h"
#include "mot_alg.h"

void user_exit(void)
{
    printf("user_exit\r\n");
}

int main(int argc, char **argv)
{
    // Initialize SDK.
    if (algsdk_init() < 0)
        return -1;
    
    // Register user exit callback function.
    algsdk_user_exit_regist(user_exit);
    
    const int natom = 2;
    const unsigned int alg_id = 2001;
    const char *desc = "multi-object tracking";    
    void *(*atom_list[natom])(void *) = {
        mot_alg_ability_data_fetch,
        mot_alg_ability_inference};
    
    // Register user algorithms.
    algsdk_alg_ability_regist(
        alg_id,
        desc,
        mot_alg_ability_init,
        mot_alg_ability_exit,
        natom,
        atom_list);

    while (1)
        sleep(1);
    
    return 0;
}