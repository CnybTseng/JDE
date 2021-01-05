#include <map>
#include <cfloat>
#include <ostream>
#include <iostream>
#include <json/json.h>
#include <opencv2/opencv.hpp>

#include "com_sihan_system_jni_utils_TrackMerge.h"
#include "lapjv.h"
#include "homography.h"

static std::map<std::string, cv::Mat> homography = {
    {"20-21", cv::Mat(3, 3, CV_32F, (void*)h20_21)},
    {"21-53", cv::Mat(3, 3, CV_32F, (void*)h21_53)},
    {"53-52", cv::Mat(3, 3, CV_32F, (void*)h53_52)}
};

static jstring charTojstring( JNIEnv* env,const char* str )
{
	jclass strClass = env->FindClass( "Ljava/lang/String;"); 
	jmethodID ctorID = env->GetMethodID( strClass, "<init>", 
		"([BLjava/lang/String;)V"); 

	if (env->ExceptionCheck() == JNI_TRUE || str == NULL)
	{
		env->ExceptionDescribe();
		env->ExceptionClear();
		printf("nativeTojstring函数转换时,str为空/n");
		return NULL;
	} 

	jbyteArray bytes = env->NewByteArray( strlen(str)); 
	//如果str为空则抛出异常给jvm

	env->SetByteArrayRegion( bytes, 0,  strlen(str), (jbyte*)str); 
	//jstring encoding = env->NewStringUTF( "GBK"); 
	jstring encoding = env->NewStringUTF( "UTF8"); 
	jstring strRtn = (jstring)env->NewObject( strClass, ctorID, bytes, 
		encoding);
	//释放str内存
	// free(str);
	return strRtn;
}

static void LinearAssignment(const cv::Mat &cost, float cost_limit,
    std::map<int, int> &matches, std::vector<int> &mismatch_row,
    std::vector<int> &mismatch_col)
{
    matches.clear();
    mismatch_row.clear();    
    mismatch_col.clear();
    
    if (cost.empty())
    {
        for (int i = 0; i < cost.rows; ++i)
            mismatch_row.emplace_back(i);
        for (int i = 0; i < cost.cols; ++i)
            mismatch_col.emplace_back(i);
        return;
    }
    
    float opt;
    cv::Mat x(cost.rows, 1, CV_32S);
    cv::Mat y(cost.cols, 1, CV_32S);
    
    bool ret = mot::LAPJV::instance()->solve((float *)cost.data, cost.rows, cost.cols, &opt,
        (int *)x.data, (int *)y.data, true, cost_limit);
    if (!ret)
        return;
    
    for (int i = 0; i < x.rows; ++i)
    {
        int j = *x.ptr<int>(i);
        if (j >= 0)
            matches.insert({i, j});
        else
            mismatch_row.emplace_back(i);
    }
    
    for (int j = 0; j < y.rows; ++j)
    {
        int i = *y.ptr<int>(j);
        if (i < 0)
            mismatch_col.emplace_back(j);
    }
}

static bool ParseFootprintFromJson(JNIEnv *env, jstring tracks,
    Json::Value &jtracks, std::vector<std::vector<cv::Mat>> &footprint, cv::Mat H=cv::Mat())
{
    std::string stracks = env->GetStringUTFChars(tracks, 0);
    JSONCPP_STRING err;
    Json::CharReaderBuilder builder;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(stracks.c_str(), stracks.c_str() + stracks.length(), &jtracks, &err)) {
        std::cout << "parse json string fail\n" << std::endl;
        return false;
    }

    for (int i = 0; i < jtracks.size(); ++i) {
        std::vector<cv::Mat> fps;
        for (int j = 0; j < jtracks[i]["rects"].size(); ++j) {
            cv::Mat fp = cv::Mat::ones(3, 1, CV_32F);
            int x = std::stoi(jtracks[i]["rects"][j]["x"].asString(), nullptr);
            int y = std::stoi(jtracks[i]["rects"][j]["y"].asString(), nullptr);
            int w = std::stoi(jtracks[i]["rects"][j]["width"].asString(), nullptr);
            int h = std::stoi(jtracks[i]["rects"][j]["height"].asString(), nullptr);
            if (x > 0 || y > 0 || w > 0 || h > 0) {
                *fp.ptr<float>(0) = x + w * 0.5f;
                *fp.ptr<float>(1) = y + h;
                if (!H.empty()) {
                    fp = H * fp;
                    *fp.ptr<float>(0) /= *fp.ptr<float>(2);
                    *fp.ptr<float>(1) /= *fp.ptr<float>(2);
                }
            } else {
                *fp.ptr<float>(0) = 0;
                *fp.ptr<float>(1) = 0;
            }
            fps.emplace_back(fp);
        }
        footprint.emplace_back(fps);
    }
    
    return true;
}

JNIEXPORT jstring JNICALL Java_com_sihan_system_jni_utils_TrackMerge_get_1registered_1channels
  (JNIEnv *env, jobject obj)
{
    Json::Value keys;
    int i = 0;
    for (auto &iter : homography) {
        keys[i++] = iter.first;
    }
    return charTojstring(env, keys.toStyledString().c_str());
}

JNIEXPORT jstring JNICALL Java_com_sihan_system_jni_utils_TrackMerge_merge_1track
  (JNIEnv *env, jobject obj, jstring tracks1, jstring tracks2, jint channel1,
  jint channel2, jint cost_thresh)
{
    Json::Value tracks;
    if (NULL == tracks1 || NULL == tracks2) {
        std::cout << "empty track!" << std::endl;
        return charTojstring(env, tracks.toStyledString().c_str());
    }
    
    std::ostringstream oss;
    oss << channel1 << "-" << channel2;
    const std::string &key = oss.str();
    
    // Requested channels have not been registered yet.
    if (homography.end() == homography.find(key)) {
        std::cout << "channels have not been registered yet" << std::endl;
        return charTojstring(env, tracks.toStyledString().c_str());
    }
    
    // Look up homography matrix.
    cv::Mat &h = homography[key];
    Json::Value jtracks1, jtracks2;

    // Parse locations from JSON string.
    std::vector<std::vector<cv::Mat>> footprint1;
    std::vector<std::vector<cv::Mat>> footprint2;
    if (!ParseFootprintFromJson(env, tracks1, jtracks1, footprint1, h) ||
        !ParseFootprintFromJson(env, tracks2, jtracks2, footprint2)) {
        return charTojstring(env, tracks.toStyledString().c_str());
    }
    
    if (0 == footprint1.size() || 0 == footprint2.size()) {
        std::cout << "empty track!" << std::endl;
        return charTojstring(env, tracks.toStyledString().c_str());
    }

    // Calculate cost matrix.    
    cv::Mat cost(footprint1.size(), footprint2.size(), CV_32F);
    for (int i = 0; i < footprint1.size(); ++i) {
        for (int j = 0; j < footprint2.size(); ++j) {
            float dists = 0;
            float num_align = 0;
            // Overlap track length from now to past.
            int overlap_len = std::min<size_t>(footprint1[i].size(), footprint2[j].size());
            for (int k = 0; k < overlap_len; ++k) {
                cv::Mat &fp1 = footprint1[i][k];
                cv::Mat &fp2 = footprint2[j][k];
                // Ignore track lost footprint.
                if ((0 == *fp1.ptr<float>(0) && 0 == *fp1.ptr<float>(1)) ||
                    (0 == *fp2.ptr<float>(0) && 0 == *fp2.ptr<float>(1))) {
                    break;
                }
                float dx = *fp1.ptr<float>(0) - *fp2.ptr<float>(0);
                float dy = *fp1.ptr<float>(1) - *fp2.ptr<float>(1);
                dists += sqrt(dx * dx + dy * dy);
                ++num_align;
                // if (k < 25)
                //     fprintf(stdout, ">>> %d %d %f\n", i, j, sqrt(dx * dx + dy * dy));
            }
            if (num_align > 0)
                dists = dists / num_align;
            else
                dists = FLT_MAX;
            *cost.ptr<float>(i, j) = dists;
        }
    }

    // Linear assignment.
    std::map<int, int> matches;
    std::vector<int> mismatch_row;
    std::vector<int> mismatch_col;
    LinearAssignment(cost, static_cast<float>(cost_thresh), matches, mismatch_row, mismatch_col);
    
    // Construct merged pair.
    // int cnt = 0;
    // std::map<int, int>::iterator iter0;
    // for (iter0 = matches.begin(); iter0 != matches.end(); iter0++) {
    //     tracks[cnt]["track1"] = jtracks1[iter0->first]["identifier"];
    //     tracks[cnt]["track2"] = jtracks2[iter0->second]["identifier"];
    //     cnt++;
    // }
    
    // Merge overlap tracks.
    int merged_id = 0;
    std::map<int, int>::iterator iter;
    for (iter = matches.begin(); iter != matches.end(); iter++) {
        Json::Value rects;
        tracks[merged_id]["identifier"] = std::to_string(merged_id);
        tracks[merged_id]["category"] = jtracks2[iter->second]["category"];
        tracks[merged_id]["rects"] = rects;
        std::vector<cv::Mat> &fps1 = footprint1[iter->first];
        std::vector<cv::Mat> &fps2 = footprint2[iter->second];
        
        // Give priority to the second channel track.
        for (int i = 0; i < fps2.size(); ++i) {
            tracks[merged_id]["rects"][i] = jtracks2[iter->second]["rects"][i];
        }
        
        // Only if fps1 is longer than fps2.
        for (int i = fps2.size(); i < fps1.size(); ++i) {
            int x = static_cast<int>(round(*fps1[i].ptr<float>(0)));
            int y = static_cast<int>(round(*fps1[i].ptr<float>(1)));
            tracks[merged_id]["rects"][i]["x"] = std::to_string(x);
            tracks[merged_id]["rects"][i]["y"] = std::to_string(y);
            tracks[merged_id]["rects"][i]["width"] = 0;
            tracks[merged_id]["rects"][i]["height"] = 0;
        }
        
        merged_id++;
    }
    
    // Append non-overlap tracks.
    for (int i = 0; i < mismatch_row.size(); ++i) {
        Json::Value rects;
        tracks[merged_id]["identifier"] = std::to_string(merged_id);
        tracks[merged_id]["category"] = jtracks1[mismatch_row[i]]["category"];
        tracks[merged_id]["rects"] = rects;
        std::vector<cv::Mat> &fps1 = footprint1[mismatch_row[i]];
        for (int j = 0; j < fps1.size(); ++j) {
            int x = static_cast<int>(round(*fps1[j].ptr<float>(0)));
            int y = static_cast<int>(round(*fps1[j].ptr<float>(1)));
            tracks[merged_id]["rects"][j]["x"] = std::to_string(x);
            tracks[merged_id]["rects"][j]["y"] = std::to_string(y);
            tracks[merged_id]["rects"][j]["width"] = std::to_string(0);
            tracks[merged_id]["rects"][j]["height"] = std::to_string(0);
        }
        ++merged_id;
    }
    
    for (int j = 0; j < mismatch_col.size(); ++j) {
        tracks[merged_id] = jtracks2[mismatch_col[j]];
        tracks[merged_id]["identifier"] = std::to_string(merged_id);
        ++merged_id;
    }

    // Debug.
    // FILE *fp = fopen("warp.txt", "w");
    // for (int k = 0; k < footprint1.size(); ++k) {
    //     cv::Mat &loc = footprint1[k][0];
    //     fprintf(fp, "%f %f\n", *loc.ptr<float>(0), *loc.ptr<float>(1));
    // }
    // fclose(fp);
    // std::cout << "cost=\n" << cost << std::endl;
    // for (iter = matches.begin(); iter != matches.end(); iter++) {
    //     std::cout << "match: " << jtracks1[iter->first]["identifier"] <<
    //         "," << jtracks2[iter->second]["identifier"] << "||" <<
    //         iter->first << ", " << iter->second << std::endl;
    // }
    // for (int i = 0; i < mismatch_row.size(); ++i) {
    //     std::cout << "row mismatch: " << mismatch_row[i] << std::endl;
    // }    
    // for (int i = 0; i < mismatch_col.size(); ++i) {
    //     std::cout << "col mismatch: " << mismatch_col[i] << std::endl;
    // }

    return charTojstring(env, tracks.toStyledString().c_str());
}