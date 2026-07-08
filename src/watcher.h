#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class SourceWatcher {
public:
    SourceWatcher();
    ~SourceWatcher();

    SourceWatcher(const SourceWatcher&) = delete;
    SourceWatcher& operator=(const SourceWatcher&) = delete;

    void watch_kernel(const std::string& kernel_name,
                      const std::string& source_dir);
    std::vector<std::string> poll();

private:
    int inotify_fd_;
    std::unordered_map<int, std::string> wd_to_kernel_;
};
