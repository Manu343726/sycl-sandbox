#pragma once
#include "sandbox_api.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

struct KernelHandle {
    void*       handle     = nullptr;
    KernelDesc  desc       = {};
    int         generation = 0;
    std::string name;
    std::string path;
};

class KernelLibrary {
public:
    KernelLibrary(std::string build_dir);
    ~KernelLibrary();

    KernelHandle* load(const std::string& kernel_name);
    void          unload(KernelHandle* kh);
    bool          rebuild(const std::string& kernel_name);
    KernelHandle* active(const std::string& kernel_name) const;

private:
    std::string build_dir_;
    std::unordered_map<std::string, std::unique_ptr<KernelHandle>> handles_;
    std::unordered_map<std::string, KernelHandle*> active_;
    int next_generation_ = 1;
};
