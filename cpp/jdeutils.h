#ifndef JDEUTILS_H
#define JDEUTILS_H

#define __check_error_goto(expre, label, fmt, ...)  \
do {                                                \
    if (expre)                                      \
    {                                               \
        fprintf(stderr, fmt, ## __VA_ARGS__);       \
        goto label;                                 \
    }                                               \
} while (0)

#define check_error_goto(expre, label, fmt, ...)    \
    __check_error_goto(expre, label, "%s:%d" fmt, __FILE__, __LINE__, ## __VA_ARGS__)

#endif  // JDEUTILS_H