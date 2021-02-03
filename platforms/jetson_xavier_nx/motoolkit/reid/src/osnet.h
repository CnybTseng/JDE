/********************************************************************
 *
 * @file osnet.h
 * @brief The TensorRT implementation of OSNet.
 * @details
 *        Paper: https://arxiv.org/abs/1905.00953
 *        Model weights: https://github.com/KaiyangZhou/deep-person-reid
 * @author CnybTseng
 * @version 1.0.0
 * @date Jan. 30, 2021
 *
 *******************************************************************/

#ifndef OSNET_H_
#define OSNET_H_

#include <memory>
#include <string>
#include <NvInfer.h>
#include "utils.h"

namespace reid {

/**
 * @brief class OSNet.
 */
class OSNet
{
public:
    OSNet();
    ~OSNet();
public:
    /**
     * @brief OSNet initialization.
     * @details
     *  The weight file must be generated from models in
     *  https://github.com/KaiyangZhou/deep-person-reid
     *
     * @param engine_path Engine file path.
     * @param weight_path Weight file (.wts) path.
     * @param beta OSNet width multipiler. Options are 25, 50, 75, and 100.
     * @param gamma OSNet input size multipiler. Options are 25, 50, 75 and 100.
     *
     * @return True if success, false otherwise.
     */
    bool init(std::string engine_path, std::string weight_path="",
        int beta=100, int gamma=100);
    /**
     * @brief OSNet deinitialization.
     *
     * @return True if success, false otherwise.
     */
    bool deinit(void);
    /**
     * @brief OSNet forward.
     *
     * @param in Float type input buffer.
     * @param out Float type output buffer.
     * @param batch_size Input and output batch size.
     *
     * @return True if success, false otherwise.
     */
    bool forward(std::shared_ptr<float> in, std::shared_ptr<float> &out,
        int batch_size);
    /**
     * @brief Get input dimension.
     *
     * @return The input dimension.
     */
    mot::DimsX get_input_dim() const;
    /**
     * @brief Get output dimension.
     *
     * @return The output dimension.
     */
    mot::DimsX get_output_dim() const;
    /**
     * @brief Get maximum batch size supported.
     *
     * @return The maximum batch size supported.
     */
    int get_max_batch_size() const;
private:
    //! Keep details in hidden, hahaha.
    class Impl;
    std::unique_ptr<Impl> impl;
};

}   // namespace reid

#endif