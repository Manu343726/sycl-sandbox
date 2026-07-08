#include "watcher.h"
#include <sys/inotify.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>

SourceWatcher::SourceWatcher() {
    inotify_fd_ = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
    if (inotify_fd_ < 0)
        perror("inotify_init1");
}

SourceWatcher::~SourceWatcher() {
    if (inotify_fd_ >= 0)
        close(inotify_fd_);
}

void SourceWatcher::watch_kernel(const std::string& kernel_name,
                                 const std::string& source_dir) {
    int wd = inotify_add_watch(inotify_fd_, source_dir.c_str(),
                               IN_CLOSE_WRITE | IN_MOVED_TO);
    if (wd < 0) {
        perror(("inotify_add_watch " + source_dir).c_str());
        return;
    }
    wd_to_kernel_[wd] = kernel_name;
}

std::vector<std::string> SourceWatcher::poll() {
    std::vector<std::string> dirty;
    alignas(struct inotify_event) char buf[4096];
    ssize_t len = read(inotify_fd_, buf, sizeof(buf));
    if (len <= 0)
        return dirty;

    for (char* p = buf; p < buf + len; ) {
        auto* ev = reinterpret_cast<struct inotify_event*>(p);
        if (ev->len > 0) {
            const char* ext = strrchr(ev->name, '.');
            if (ext && (!strcmp(ext, ".cpp") || !strcmp(ext, ".h") ||
                        !strcmp(ext, ".hpp") || !strcmp(ext, ".cl"))) {
                auto it = wd_to_kernel_.find(ev->wd);
                if (it != wd_to_kernel_.end()) {
                    bool dup = false;
                    for (auto& k : dirty)
                        if (k == it->second) { dup = true; break; }
                    if (!dup)
                        dirty.push_back(it->second);
                }
            }
        }
        p += sizeof(struct inotify_event) + ev->len;
    }
    return dirty;
}
