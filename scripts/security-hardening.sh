#!/bin/bash

# Security Hardening Script for Media Server
# Implements security best practices and policies

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║          Media Server Security Hardening Script                  ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print messages
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to generate secure passwords
generate_secure_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then 
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Function to create security audit log
create_audit_log() {
    local log_dir="/var/log/media-server-security"
    mkdir -p "$log_dir"
    
    local log_file="$log_dir/security-hardening-$(date +%Y%m%d_%H%M%S).log"
    
    cat > "$log_file" << EOF
Media Server Security Hardening Log
===================================
Date: $(date)
User: $(whoami)
System: $(uname -a)
===================================

Security measures applied:
EOF
    
    echo "$log_file"
}

# Function to secure Docker daemon
secure_docker() {
    print_info "Securing Docker daemon..."
    
    # Create Docker security options
    cat > /etc/docker/daemon.json << 'EOF'
{
  "icc": false,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "userland-proxy": false,
  "no-new-privileges": true,
  "selinux-enabled": true,
  "userns-remap": "default",
  "live-restore": true,
  "init": true,
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  },
  "seccomp-profile": "/etc/docker/seccomp.json"
}
EOF
    
    # Create custom seccomp profile
    cat > /etc/docker/seccomp.json << 'EOF'
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": [
    "SCMP_ARCH_X86_64",
    "SCMP_ARCH_X86",
    "SCMP_ARCH_X32"
  ],
  "syscalls": [
    {
      "names": [
        "accept",
        "accept4",
        "access",
        "alarm",
        "bind",
        "brk",
        "chdir",
        "chmod",
        "chown",
        "clock_getres",
        "clock_gettime",
        "clock_nanosleep",
        "clone",
        "close",
        "connect",
        "copy_file_range",
        "dup",
        "dup2",
        "dup3",
        "epoll_create",
        "epoll_create1",
        "epoll_ctl",
        "epoll_ctl_old",
        "epoll_pwait",
        "epoll_wait",
        "epoll_wait_old",
        "eventfd",
        "eventfd2",
        "execve",
        "execveat",
        "exit",
        "exit_group",
        "faccessat",
        "fadvise64",
        "fadvise64_64",
        "fallocate",
        "fanotify_mark",
        "fchdir",
        "fchmod",
        "fchmodat",
        "fchown",
        "fchown32",
        "fchownat",
        "fcntl",
        "fcntl64",
        "fdatasync",
        "fgetxattr",
        "flistxattr",
        "flock",
        "fork",
        "fremovexattr",
        "fsetxattr",
        "fstat",
        "fstat64",
        "fstatat64",
        "fstatfs",
        "fstatfs64",
        "fsync",
        "ftruncate",
        "ftruncate64",
        "futex",
        "futimesat",
        "getcpu",
        "getcwd",
        "getdents",
        "getdents64",
        "getegid",
        "getegid32",
        "geteuid",
        "geteuid32",
        "getgid",
        "getgid32",
        "getgroups",
        "getgroups32",
        "getitimer",
        "getpeername",
        "getpgid",
        "getpgrp",
        "getpid",
        "getppid",
        "getpriority",
        "getrandom",
        "getresgid",
        "getresgid32",
        "getresuid",
        "getresuid32",
        "getrlimit",
        "get_robust_list",
        "getrusage",
        "getsid",
        "getsockname",
        "getsockopt",
        "get_thread_area",
        "gettid",
        "gettimeofday",
        "getuid",
        "getuid32",
        "getxattr",
        "inotify_add_watch",
        "inotify_init",
        "inotify_init1",
        "inotify_rm_watch",
        "io_cancel",
        "ioctl",
        "io_destroy",
        "io_getevents",
        "ioprio_get",
        "ioprio_set",
        "io_setup",
        "io_submit",
        "ipc",
        "kill",
        "lchown",
        "lchown32",
        "lgetxattr",
        "link",
        "linkat",
        "listen",
        "listxattr",
        "llistxattr",
        "lremovexattr",
        "lseek",
        "lsetxattr",
        "lstat",
        "lstat64",
        "madvise",
        "memfd_create",
        "mincore",
        "mkdir",
        "mkdirat",
        "mknod",
        "mknodat",
        "mlock",
        "mlock2",
        "mlockall",
        "mmap",
        "mmap2",
        "mprotect",
        "mq_getsetattr",
        "mq_notify",
        "mq_open",
        "mq_timedreceive",
        "mq_timedsend",
        "mq_unlink",
        "mremap",
        "msgctl",
        "msgget",
        "msgrcv",
        "msgsnd",
        "msync",
        "munlock",
        "munlockall",
        "munmap",
        "nanosleep",
        "newfstatat",
        "open",
        "openat",
        "pause",
        "pipe",
        "pipe2",
        "poll",
        "ppoll",
        "prctl",
        "pread64",
        "preadv",
        "prlimit64",
        "pselect6",
        "pwrite64",
        "pwritev",
        "read",
        "readahead",
        "readlink",
        "readlinkat",
        "readv",
        "recv",
        "recvfrom",
        "recvmmsg",
        "recvmsg",
        "remap_file_pages",
        "removexattr",
        "rename",
        "renameat",
        "renameat2",
        "restart_syscall",
        "rmdir",
        "rt_sigaction",
        "rt_sigpending",
        "rt_sigprocmask",
        "rt_sigqueueinfo",
        "rt_sigreturn",
        "rt_sigsuspend",
        "rt_sigtimedwait",
        "rt_tgsigqueueinfo",
        "sched_getaffinity",
        "sched_getattr",
        "sched_getparam",
        "sched_get_priority_max",
        "sched_get_priority_min",
        "sched_getscheduler",
        "sched_rr_get_interval",
        "sched_setaffinity",
        "sched_setattr",
        "sched_setparam",
        "sched_setscheduler",
        "sched_yield",
        "seccomp",
        "select",
        "semctl",
        "semget",
        "semop",
        "semtimedop",
        "send",
        "sendfile",
        "sendfile64",
        "sendmmsg",
        "sendmsg",
        "sendto",
        "setfsgid",
        "setfsgid32",
        "setfsuid",
        "setfsuid32",
        "setgid",
        "setgid32",
        "setgroups",
        "setgroups32",
        "setitimer",
        "setpgid",
        "setpriority",
        "setregid",
        "setregid32",
        "setresgid",
        "setresgid32",
        "setresuid",
        "setresuid32",
        "setreuid",
        "setreuid32",
        "setrlimit",
        "set_robust_list",
        "setsid",
        "setsockopt",
        "set_thread_area",
        "set_tid_address",
        "setuid",
        "setuid32",
        "setxattr",
        "shmat",
        "shmctl",
        "shmdt",
        "shmget",
        "shutdown",
        "sigaltstack",
        "signalfd",
        "signalfd4",
        "sigreturn",
        "socket",
        "socketcall",
        "socketpair",
        "splice",
        "stat",
        "stat64",
        "statfs",
        "statfs64",
        "symlink",
        "symlinkat",
        "sync",
        "sync_file_range",
        "syncfs",
        "sysinfo",
        "syslog",
        "tee",
        "tgkill",
        "time",
        "timer_create",
        "timer_delete",
        "timerfd_create",
        "timerfd_gettime",
        "timerfd_settime",
        "timer_getoverrun",
        "timer_gettime",
        "timer_settime",
        "times",
        "tkill",
        "truncate",
        "truncate64",
        "ugetrlimit",
        "umask",
        "uname",
        "unlink",
        "unlinkat",
        "utime",
        "utimensat",
        "utimes",
        "vfork",
        "vmsplice",
        "wait4",
        "waitid",
        "waitpid",
        "write",
        "writev"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
EOF
    
    # Set up user namespace
    echo "dockremap:231072:65536" >> /etc/subuid
    echo "dockremap:231072:65536" >> /etc/subgid
    
    print_success "Docker daemon secured"
}

# Function to create AppArmor profiles
create_apparmor_profiles() {
    print_info "Creating AppArmor profiles..."
    
    if command -v aa-status &> /dev/null; then
        # Create AppArmor profile for containers
        cat > /etc/apparmor.d/docker-media-server << 'EOF'
#include <tunables/global>

profile docker-media-server flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>

  network,
  capability,

  file,
  umount,

  deny @{PROC}/* w,
  deny @{PROC}/*.* w,
  deny @{PROC}/sys/kernel/** w,
  deny @{PROC}/sysrq-trigger rwklx,
  deny @{PROC}/mem rwklx,
  deny @{PROC}/kmem rwklx,
  deny @{PROC}/kcore rwklx,

  deny mount,

  deny /sys/** w,
  
  # Allow necessary paths
  /var/lib/docker/** rw,
  /var/run/docker.sock rw,
  /etc/localtime r,
  /etc/hosts r,
  /etc/hostname r,
  /etc/resolv.conf r,
}
EOF
        
        apparmor_parser -r /etc/apparmor.d/docker-media-server
        print_success "AppArmor profiles created"
    else
        print_warning "AppArmor not available, skipping profile creation"
    fi
}

# Function to set up firewall rules
setup_firewall() {
    print_info "Setting up firewall rules..."
    
    if command -v ufw &> /dev/null; then
        # Reset firewall
        ufw --force reset
        
        # Default policies
        ufw default deny incoming
        ufw default allow outgoing
        
        # Allow SSH
        ufw allow 22/tcp comment "SSH"
        
        # Allow HTTP/HTTPS
        ufw allow 80/tcp comment "HTTP"
        ufw allow 443/tcp comment "HTTPS"
        
        # Media server ports (only from local network)
        local local_subnet="192.168.0.0/16"
        
        # Jellyfin
        ufw allow from $local_subnet to any port 8096 comment "Jellyfin HTTP"
        ufw allow from $local_subnet to any port 8920 comment "Jellyfin HTTPS"
        ufw allow from $local_subnet to any port 1900/udp comment "Jellyfin DLNA"
        ufw allow from $local_subnet to any port 7359/udp comment "Jellyfin Discovery"
        
        # Other services (local only)
        ufw allow from $local_subnet to any port 3000 comment "Homepage"
        ufw allow from $local_subnet to any port 9000 comment "Portainer"
        
        # Enable firewall
        ufw --force enable
        
        print_success "Firewall rules configured"
    else
        print_warning "UFW not installed, skipping firewall setup"
    fi
}

# Function to create fail2ban configuration
setup_fail2ban() {
    print_info "Setting up Fail2ban..."
    
    if command -v fail2ban-client &> /dev/null; then
        # Create jail configuration
        cat > /etc/fail2ban/jail.d/media-server.conf << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = root@localhost
sendername = Fail2Ban
action = %(action_mwl)s

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[traefik-auth]
enabled = true
port = http,https
filter = traefik-auth
logpath = /var/log/traefik/access.log
maxretry = 5

[jellyfin-auth]
enabled = true
port = 8096,8920
filter = jellyfin-auth
logpath = /var/log/jellyfin/*.log
maxretry = 5

[docker-unauthorized]
enabled = true
port = 2375,2376
filter = docker-unauthorized
logpath = /var/log/docker.log
maxretry = 2
EOF
        
        # Create filter for Traefik
        cat > /etc/fail2ban/filter.d/traefik-auth.conf << 'EOF'
[Definition]
failregex = ^<HOST> .* 401 .*$
ignoreregex =
EOF
        
        # Create filter for Jellyfin
        cat > /etc/fail2ban/filter.d/jellyfin-auth.conf << 'EOF'
[Definition]
failregex = ^.*Authentication request for .* has been denied \(IP: <HOST>\).*$
ignoreregex =
EOF
        
        # Create filter for Docker
        cat > /etc/fail2ban/filter.d/docker-unauthorized.conf << 'EOF'
[Definition]
failregex = ^.*\[error\].*unauthorized: authentication required.*<HOST>.*$
ignoreregex =
EOF
        
        # Restart fail2ban
        systemctl restart fail2ban
        
        print_success "Fail2ban configured"
    else
        print_warning "Fail2ban not installed, skipping configuration"
    fi
}

# Function to harden system kernel
harden_kernel() {
    print_info "Hardening kernel parameters..."
    
    cat > /etc/sysctl.d/99-security-hardening.conf << 'EOF'
# Kernel hardening parameters

# Disable kernel modules loading
kernel.modules_disabled = 1

# Restrict kernel logs
kernel.dmesg_restrict = 1

# Restrict ptrace scope
kernel.yama.ptrace_scope = 2

# Enable ExecShield
kernel.exec-shield = 1

# Randomize memory regions
kernel.randomize_va_space = 2

# Hide kernel pointers
kernel.kptr_restrict = 2

# Disable magic SysRq key
kernel.sysrq = 0

# Disable core dumps
fs.suid_dumpable = 0

# Network security
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.tcp_syncookies = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# IPv6 security
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
EOF
    
    # Apply settings
    sysctl -p /etc/sysctl.d/99-security-hardening.conf
    
    print_success "Kernel parameters hardened"
}

# Function to set up audit logging
setup_auditd() {
    print_info "Setting up audit logging..."
    
    if command -v auditctl &> /dev/null; then
        # Create audit rules
        cat > /etc/audit/rules.d/media-server.rules << 'EOF'
# Media Server Audit Rules

# Log all commands
-a exit,always -F arch=b64 -S execve
-a exit,always -F arch=b32 -S execve

# Log file access
-w /etc/passwd -p wa -k passwd_changes
-w /etc/group -p wa -k group_changes
-w /etc/shadow -p wa -k shadow_changes
-w /etc/sudoers -p wa -k sudoers_changes

# Log Docker access
-w /usr/bin/docker -p x -k docker_commands
-w /var/lib/docker -p wa -k docker_files
-w /etc/docker -p wa -k docker_config

# Log system calls
-a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time_change
-a always,exit -F arch=b32 -S adjtimex -S settimeofday -S stime -k time_change
-a always,exit -F arch=b64 -S mount -S umount2 -k mount_changes
-a always,exit -F arch=b32 -S mount -S umount -S umount2 -k mount_changes

# Log network connections
-a always,exit -F arch=b64 -S socket -S connect -k network_connections
-a always,exit -F arch=b32 -S socket -S connect -k network_connections
EOF
        
        # Reload audit rules
        augenrules --load
        systemctl restart auditd
        
        print_success "Audit logging configured"
    else
        print_warning "Auditd not installed, skipping audit configuration"
    fi
}

# Function to secure SSH
secure_ssh() {
    print_info "Securing SSH configuration..."
    
    if [ -f /etc/ssh/sshd_config ]; then
        # Backup original config
        cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup
        
        # Apply secure settings
        cat > /etc/ssh/sshd_config.d/99-security-hardening.conf << 'EOF'
# SSH Security Hardening

# Disable root login
PermitRootLogin no

# Use key-based authentication only
PasswordAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes

# Limit users
AllowUsers deploy

# Security settings
Protocol 2
StrictModes yes
IgnoreRhosts yes
HostbasedAuthentication no
PermitEmptyPasswords no
X11Forwarding no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
LoginGraceTime 30
MaxStartups 10:30:60
Banner /etc/ssh/banner

# Use strong ciphers
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com,aes256-ctr,aes192-ctr,aes128-ctr
MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com,umac-128-etm@openssh.com
KexAlgorithms curve25519-sha256,curve25519-sha256@libssh.org,diffie-hellman-group16-sha512,diffie-hellman-group18-sha512
EOF
        
        # Create SSH banner
        cat > /etc/ssh/banner << 'EOF'
*****************************************************************************
                            AUTHORIZED ACCESS ONLY
*****************************************************************************
This system is for authorized use only. All activity is monitored and logged.
Unauthorized access is strictly prohibited and will be prosecuted.
*****************************************************************************
EOF
        
        # Restart SSH
        systemctl restart sshd
        
        print_success "SSH configuration secured"
    else
        print_warning "SSH config not found, skipping SSH hardening"
    fi
}

# Function to create security monitoring script
create_security_monitor() {
    print_info "Creating security monitoring script..."
    
    cat > /usr/local/bin/security-monitor.sh << 'EOF'
#!/bin/bash

# Security monitoring script

LOG_FILE="/var/log/media-server-security/monitor.log"

log_event() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check for failed login attempts
failed_logins=$(journalctl -u sshd --since "1 hour ago" | grep -c "Failed password")
if [ "$failed_logins" -gt 0 ]; then
    log_event "WARNING: $failed_logins failed SSH login attempts in the last hour"
fi

# Check for unauthorized Docker access
docker_unauthorized=$(docker events --since "1h" --filter "event=die" --format "{{.Actor.Attributes.exitCode}}" | grep -c "125\|126")
if [ "$docker_unauthorized" -gt 0 ]; then
    log_event "WARNING: $docker_unauthorized unauthorized Docker access attempts"
fi

# Check for suspicious processes
suspicious_procs=$(ps aux | grep -E "(nc|netcat|ncat|socat)" | grep -v grep | wc -l)
if [ "$suspicious_procs" -gt 0 ]; then
    log_event "WARNING: Suspicious network processes detected"
fi

# Check for open ports
open_ports=$(ss -tuln | grep LISTEN | wc -l)
log_event "INFO: $open_ports open ports detected"

# Check disk usage
disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$disk_usage" -gt 90 ]; then
    log_event "WARNING: Disk usage critical at $disk_usage%"
fi

# Check for rootkits
if command -v rkhunter &> /dev/null; then
    rkhunter --check --skip-keypress --report-warnings-only >> "$LOG_FILE" 2>&1
fi
EOF
    
    chmod +x /usr/local/bin/security-monitor.sh
    
    # Create cron job
    cat > /etc/cron.d/security-monitor << 'EOF'
# Run security monitoring every hour
0 * * * * root /usr/local/bin/security-monitor.sh
EOF
    
    print_success "Security monitoring script created"
}

# Function to generate security report
generate_security_report() {
    local report_file="/var/log/media-server-security/security-report-$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Media Server Security Report
============================
Generated: $(date)

System Information:
------------------
Hostname: $(hostname)
Kernel: $(uname -r)
Uptime: $(uptime)

Security Status:
---------------
EOF
    
    # Check service status
    echo "Service Status:" >> "$report_file"
    for service in docker fail2ban auditd ufw; do
        if systemctl is-active --quiet $service; then
            echo "  ✓ $service is running" >> "$report_file"
        else
            echo "  ✗ $service is not running" >> "$report_file"
        fi
    done
    
    echo "" >> "$report_file"
    echo "Docker Security:" >> "$report_file"
    docker info 2>/dev/null | grep -E "(Storage Driver|Logging Driver|Cgroup Driver|Security Options)" >> "$report_file"
    
    echo "" >> "$report_file"
    echo "Open Ports:" >> "$report_file"
    ss -tuln | grep LISTEN >> "$report_file"
    
    echo "" >> "$report_file"
    echo "User Accounts:" >> "$report_file"
    awk -F: '$3 >= 1000 {print "  " $1 " (UID: " $3 ")"}' /etc/passwd >> "$report_file"
    
    print_success "Security report generated: $report_file"
}

# Main execution
main() {
    # Check if running as root
    check_root
    
    echo ""
    print_warning "This script will apply security hardening to your system."
    print_warning "Some changes may affect system behavior."
    echo ""
    
    read -p "Do you want to continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    echo ""
    
    # Create audit log
    LOG_FILE=$(create_audit_log)
    
    # Apply security measures
    secure_docker >> "$LOG_FILE" 2>&1
    create_apparmor_profiles >> "$LOG_FILE" 2>&1
    setup_firewall >> "$LOG_FILE" 2>&1
    setup_fail2ban >> "$LOG_FILE" 2>&1
    harden_kernel >> "$LOG_FILE" 2>&1
    setup_auditd >> "$LOG_FILE" 2>&1
    secure_ssh >> "$LOG_FILE" 2>&1
    create_security_monitor >> "$LOG_FILE" 2>&1
    
    # Generate security report
    generate_security_report
    
    echo ""
    print_success "Security hardening complete!"
    echo ""
    print_info "Review the security report for details"
    print_info "Monitor /var/log/media-server-security/ for ongoing security events"
    echo ""
    print_warning "Remember to:"
    print_warning "- Update the 'deploy' user for SSH access"
    print_warning "- Configure your firewall subnet if not 192.168.0.0/16"
    print_warning "- Review and adjust AppArmor profiles as needed"
    print_warning "- Regularly update all containers and the host system"
}

# Run main function
main "$@"