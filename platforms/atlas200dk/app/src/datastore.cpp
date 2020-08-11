#include <cstdio>
#include <dirent.h>
#include <unistd.h>
#include <fstream>
#include <c_graph.h>
#include <sys/stat.h>

#include <algorithm>

#include "datastore.h"

bool is_directory(const std::string path)
{
    struct stat st;
    if (0 != stat(path.c_str(), &st)) {
        return false;
    }
    
    if (S_ISDIR(st.st_mode)) {
        return true;
    } else {
        return false;
    }
}

void read_path_files(const std::string path, std::vector<std::string> &fvec)
{
    dirent *dent = nullptr;
    DIR *dirp = nullptr;
    if (is_directory(path)) {
        dirp = opendir(path.c_str());
        while (nullptr != (dent = readdir(dirp))) {
            if ('.' == dent->d_name[0]) {
                continue;
            }
            
            std::string fpath = path + "/" + dent->d_name;
            if (is_directory(fpath)) {
                read_path_files(fpath, fvec);
            } else {
                fvec.emplace_back(fpath);
            }
        }
    } else {
        fvec.emplace_back(path);
    }
}

void bgr2_yuv420sp_nv12(const cv::Mat &bgr, cv::Mat &nv12)
{
    cv::Mat yuv444;
    cv::cvtColor(bgr, yuv444, cv::COLOR_BGR2YUV);
    
    std::vector<cv::Mat> yuv(3);
    cv::split(yuv444, yuv);
    
    const int32_t w = bgr.cols;
    const int32_t h = bgr.rows;
    const int32_t hw = w >> 1;
    const int32_t hh = h >> 1;
    
    uint8_t *sy = yuv[0].data;
    uint8_t *su = yuv[1].data;
    uint8_t *sv = yuv[2].data;
    
    uint8_t *dy = nv12.data;
    uint8_t *du = dy + w * h;
    uint8_t *dv = du + 1;
    
    memcpy(dy, sy, w * h * sizeof(uint8_t));
    
    for (int32_t i = 0; i < hh; ++i) {
        for (int32_t j = 0; j < hw; ++j) {
            *du = *su;
            *dv = *sv;
            du += 2;
            dv += 2;
            su += 2;
            sv += 2;
        }
        su += w;
        sv += w;
    }
}

char *read_binary_file(std::string path, size_t &size)
{
    std::ifstream ifs;
    ifs.open(path, std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "std::ifstream::open fail\n");
        return nullptr;
    }

    std::filebuf *fbuf = ifs.rdbuf();
    size = fbuf->pubseekoff(0, std::ios::end, std::ios::in);
    fbuf->pubseekpos(0, std::ios::in);
    char *buffer = new char[size];
    if (nullptr == buffer) {
        fprintf(stderr, "HIAI_DVPP_DMalloc fail\n");
        ifs.close();
        return nullptr;
    }

    fbuf->sgetn(buffer, size);
    ifs.close();

    return buffer;
}

char *read_binary_file_for_dvpp(std::string path, size_t &size)
{
    std::ifstream ifs;
    ifs.open(path, std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "std::ifstream::open fail\n");
        return nullptr;
    }

    std::filebuf *fbuf = ifs.rdbuf();
    size = fbuf->pubseekoff(0, std::ios::end, std::ios::in);
    fbuf->pubseekpos(0, std::ios::in);
    char *buffer = static_cast<char *>(HIAI_DVPP_DMalloc(size));
    if (nullptr == buffer) {
        fprintf(stderr, "HIAI_DVPP_DMalloc fail\n");
        ifs.close();
        return nullptr;
    }

    fbuf->sgetn(buffer, size);
    ifs.close();

    return buffer;
}

void string_replace(std::string &str, const std::string &old,
    const std::string &neww)
{
    std::string::size_type pos = str.find(old);
    if (std::string::npos != pos) {
        str.replace(pos, old.length(), neww);
    } 
}

int jpeg_decode(const std::string &path, struct JpegdOut &jpegd_out)
{
    struct JpegdIn jpegd_in;
    int32_t ret = 0;
    dvppapi_ctl_msg dcm;
    IDVPPAPI *idvppapi = nullptr;
    
    FILE *fp = fopen(path.c_str(), "rb");
    if (nullptr == fp) {
        fprintf(stderr, "cann't open file %s\n", path.c_str());
        return -1;
    }
    
    fseek(fp, 0, SEEK_END);
    uint32_t len = ftell(fp);
    jpegd_in.jpegDataSize = len + 8;
    jpegd_in.isVBeforeU = false;
    fseek(fp, 0, SEEK_SET);
        
    void *raw_buf = HIAI_DVPP_DMalloc(jpegd_in.jpegDataSize);
    if (nullptr == raw_buf) {
        fprintf(stderr, "HIAI_DVPP_DMalloc fail\n");
        goto error;
    }
    
    jpegd_in.jpegData = reinterpret_cast<uint8_t *>(raw_buf);
    fread(jpegd_in.jpegData, sizeof(uint8_t), jpegd_in.jpegDataSize, fp);
    
    ret = DvppGetOutParameter((void *)(&jpegd_in), (void *)(&jpegd_out), GET_JPEGD_OUT_PARAMETER);
    if (0 != ret) {
        fprintf(stderr, "DvppGetOutParameter fail\n");
        goto error;
    }
    
    {
        fprintf(stderr, "imgWidth %u\n", jpegd_out.imgWidth);
        fprintf(stderr, "imgHeight %u\n", jpegd_out.imgHeight);
        fprintf(stderr, "imgWidthAligned %u\n", jpegd_out.imgWidthAligned);
        fprintf(stderr, "imgHeightAligned %u\n", jpegd_out.imgHeightAligned);
        fprintf(stderr, "outFormat %d\n", jpegd_out.outFormat);
    }
    
    jpegd_out.yuvData = reinterpret_cast<uint8_t *>(HIAI_DVPP_DMalloc(jpegd_out.yuvDataSize));
    if (nullptr == jpegd_out.yuvData) {
        fprintf(stderr, "HIAI_DVPP_DMalloc fail\n");
        goto error;
    }
    
    dcm.in = (void *)&jpegd_in;
    dcm.in_size = sizeof(jpegd_in);
    dcm.out = (void *)&jpegd_out;
    dcm.out_size = sizeof(jpegd_out);
    
    CreateDvppApi(idvppapi);
    if (nullptr == idvppapi) {
        fprintf(stderr, "CreateDvppApi fail\n");
        goto error;
    }
    
    ret = DvppCtl(idvppapi, DVPP_CTL_JPEGD_PROC, &dcm);
    if (0 != ret) {
        fprintf(stderr, "DvppCtl fail\n");
        goto error;
    }
    
    return 0;
    error:
    if (fp) {
        fclose(fp);
        fp = NULL;
    }
    
    if (raw_buf) {
        HIAI_DVPP_DFree(raw_buf);
        raw_buf = nullptr;
    }
    
    if (idvppapi) {
        DestroyDvppApi(idvppapi);
    }
    
    return -1;
}