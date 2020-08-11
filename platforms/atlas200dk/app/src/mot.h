#ifndef MOT_H
#define MOT_H

#include <deque>
#include <vector>
#include <string>

#if defined _WIN32 || defined __CYGWIN__
  #define MOT_HELPER_DLL_IMPORT __declspec(dllimport)
  #define MOT_HELPER_DLL_EXPORT __declspec(dllexport)
  #define MOT_HELPER_DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define MOT_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
    #define MOT_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
    #define MOT_HELPER_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define MOT_HELPER_DLL_IMPORT
    #define MOT_HELPER_DLL_EXPORT
    #define MOT_HELPER_DLL_LOCAL
  #endif
#endif

#ifdef MOT_DLL
  #ifdef MOT_DLL_EXPORTS
    #define MOT_API MOT_HELPER_DLL_EXPORT
  #else
    #define MOT_API MOT_HELPER_DLL_IMPORT
  #endif // MOT_DLL_EXPORTS
  #define MOT_LOCAL MOT_HELPER_DLL_LOCAL
#else
  #define MOT_API
  #define MOT_LOCAL
#endif // MOT_DLL

namespace mot {

struct MOT_Rect
{
    float top;
    float left;
    float bottom;
    float right;
};

enum MOT_Posture
{
    STANDING,
    LIE_DOWN,
    SQUAT
};

struct MOT_Track
{
    int identifier;
    MOT_Posture posture;
    std::string category;
    std::deque<MOT_Rect> rects;
};

typedef std::vector<MOT_Track> MOT_Result;

extern "C" MOT_API int load_mot_model(const char *cfg_path);

extern "C" MOT_API int unload_mot_model();

extern "C" MOT_API int forward_mot_model(const unsigned char *im, int width, int height, int stride, MOT_Result &result);

}   // namespace mot

#endif  // MOT_H