/**
 * @file
 * @brief BSD Sockets compatible API definitions
 *
 * An API for applications to use BSD Sockets like API.
 */

/*
 * Copyright (c) 2017-2018 Linaro Limited
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ZEPHYR_INCLUDE_NET_SOCKET_H_
#define ZEPHYR_INCLUDE_NET_SOCKET_H_

/**
 * @brief BSD Sockets compatible API
 * @defgroup bsd_sockets BSD Sockets compatible API
 * @ingroup networking
 * @{
 */

#include <sys/types.h>
#include <zephyr/types.h>
#include <net/net_ip.h>
#include <net/dns_resolve.h>
#include <net/socket_select.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct zsock_pollfd {
	int fd;
	short events;
	short revents;
};

/* Values are compatible with Linux */
#define ZSOCK_POLLIN 1
#define ZSOCK_POLLPRI 2
#define ZSOCK_POLLOUT 4
#define ZSOCK_POLLERR 8
#define ZSOCK_POLLHUP 0x10
#define ZSOCK_POLLNVAL 0x20

#define ZSOCK_MSG_PEEK 0x02
#define ZSOCK_MSG_DONTWAIT 0x40

/* Well-known values, e.g. from Linux man 2 shutdown:
 * "The constants SHUT_RD, SHUT_WR, SHUT_RDWR have the value 0, 1, 2,
 * respectively". Some software uses numeric values.
 */
#define ZSOCK_SHUT_RD 0
#define ZSOCK_SHUT_WR 1
#define ZSOCK_SHUT_RDWR 2

/** Protocol level for TLS.
 *  Here, the same socket protocol level for TLS as in Linux was used.
 */
#define SOL_TLS 282

/**
 *  @defgroup secure_sockets_options Socket options for TLS
 *  @{
 */

/** Socket option to select TLS credentials to use. It accepts and returns an
 *  array of sec_tag_t that indicate which TLS credentials should be used with
 *  specific socket.
 */
#define TLS_SEC_TAG_LIST 1
/** Write-only socket option to set hostname. It accepts a string containing
 *  the hostname (may be NULL to disable hostname verification). By default,
 *  hostname check is enforced for TLS clients.
 */
#define TLS_HOSTNAME 2
/** Socket option to select ciphersuites to use. It accepts and returns an array
 *  of integers with IANA assigned ciphersuite identifiers.
 *  If not set, socket will allow all ciphersuites available in the system
 *  (mebdTLS default behavior).
 */
#define TLS_CIPHERSUITE_LIST 3
/** Read-only socket option to read a ciphersuite chosen during TLS handshake.
 *  It returns an integer containing an IANA assigned ciphersuite identifier
 *  of chosen ciphersuite.
 */
#define TLS_CIPHERSUITE_USED 4
/** Write-only socket option to set peer verification level for TLS connection.
 *  This option accepts an integer with a peer verification level, compatible
 *  with mbedTLS values:
 *    - 0 - none
 *    - 1 - optional
 *    - 2 - required
 *
 *  If not set, socket will use mbedTLS defaults (none for servers, required
 *  for clients).
 */
#define TLS_PEER_VERIFY 5
/** Write-only socket option to set role for DTLS connection. This option
 *  is irrelevant for TLS connections, as for them role is selected based on
 *  connect()/listen() usage. By default, DTLS will assume client role.
 *  This option accepts an integer with a TLS role, compatible with
 *  mbedTLS values:
 *    - 0 - client
 *    - 1 - server
 */
#define TLS_DTLS_ROLE 6

/** @} */

struct zsock_addrinfo {
	struct zsock_addrinfo *ai_next;
	int ai_flags;
	int ai_family;
	int ai_socktype;
	int ai_protocol;
	socklen_t ai_addrlen;
	struct sockaddr *ai_addr;
	char *ai_canonname;

	struct sockaddr _ai_addr;
	char _ai_canonname[DNS_MAX_NAME_SIZE + 1];
};

__syscall int zsock_socket(int family, int type, int proto);

__syscall int zsock_close(int sock);

__syscall int zsock_shutdown(int sock, int how);

__syscall int zsock_bind(int sock, const struct sockaddr *addr,
			 socklen_t addrlen);

__syscall int zsock_connect(int sock, const struct sockaddr *addr,
			    socklen_t addrlen);

__syscall int zsock_listen(int sock, int backlog);

__syscall int zsock_accept(int sock, struct sockaddr *addr, socklen_t *addrlen);

__syscall ssize_t zsock_sendto(int sock, const void *buf, size_t len,
			       int flags, const struct sockaddr *dest_addr,
			       socklen_t addrlen);

static inline ssize_t zsock_send(int sock, const void *buf, size_t len,
				 int flags)
{
	return zsock_sendto(sock, buf, len, flags, NULL, 0);
}

__syscall ssize_t zsock_recvfrom(int sock, void *buf, size_t max_len,
				 int flags, struct sockaddr *src_addr,
				 socklen_t *addrlen);

static inline ssize_t zsock_recv(int sock, void *buf, size_t max_len,
				 int flags)
{
	return zsock_recvfrom(sock, buf, max_len, flags, NULL, NULL);
}

__syscall int zsock_fcntl(int sock, int cmd, int flags);

__syscall int zsock_poll(struct zsock_pollfd *fds, int nfds, int timeout);

int zsock_getsockopt(int sock, int level, int optname,
		     void *optval, socklen_t *optlen);

int zsock_setsockopt(int sock, int level, int optname,
		     const void *optval, socklen_t optlen);

int zsock_gethostname(char *buf, size_t len);

static inline char *zsock_inet_ntop(sa_family_t family, const void *src,
				    char *dst, size_t size)
{
	return net_addr_ntop(family, src, dst, size);
}

__syscall int zsock_inet_pton(sa_family_t family, const char *src, void *dst);

__syscall int z_zsock_getaddrinfo_internal(const char *host,
					   const char *service,
					   const struct zsock_addrinfo *hints,
					   struct zsock_addrinfo *res);

int zsock_getaddrinfo(const char *host, const char *service,
		      const struct zsock_addrinfo *hints,
		      struct zsock_addrinfo **res);

static inline void zsock_freeaddrinfo(struct zsock_addrinfo *ai)
{
	free(ai);
}

const char *zsock_gai_strerror(int errcode);

#define NI_NUMERICHOST 1
#define NI_NUMERICSERV 2
#define NI_NOFQDN 4
#define NI_NAMEREQD 8
#define NI_DGRAM 16

int zsock_getnameinfo(const struct sockaddr *addr, socklen_t addrlen,
		      char *host, socklen_t hostlen,
		      char *serv, socklen_t servlen, int flags);

#if defined(CONFIG_NET_SOCKETS_POSIX_NAMES)

#define pollfd zsock_pollfd

#if !defined(CONFIG_NET_SOCKETS_OFFLOAD)
static inline int socket(int family, int type, int proto)
{
	return zsock_socket(family, type, proto);
}

static inline int close(int sock)
{
	return zsock_close(sock);
}

static inline int shutdown(int sock, int how)
{
	return zsock_shutdown(sock, how);
}

static inline int bind(int sock, const struct sockaddr *addr, socklen_t addrlen)
{
	return zsock_bind(sock, addr, addrlen);
}

static inline int connect(int sock, const struct sockaddr *addr,
			  socklen_t addrlen)
{
	return zsock_connect(sock, addr, addrlen);
}

static inline int listen(int sock, int backlog)
{
	return zsock_listen(sock, backlog);
}

static inline int accept(int sock, struct sockaddr *addr, socklen_t *addrlen)
{
	return zsock_accept(sock, addr, addrlen);
}

static inline ssize_t send(int sock, const void *buf, size_t len, int flags)
{
	return zsock_send(sock, buf, len, flags);
}

static inline ssize_t recv(int sock, void *buf, size_t max_len, int flags)
{
	return zsock_recv(sock, buf, max_len, flags);
}

/* This conflicts with fcntl.h, so code must include fcntl.h before socket.h: */
#define fcntl zsock_fcntl

static inline ssize_t sendto(int sock, const void *buf, size_t len, int flags,
			     const struct sockaddr *dest_addr,
			     socklen_t addrlen)
{
	return zsock_sendto(sock, buf, len, flags, dest_addr, addrlen);
}

static inline ssize_t recvfrom(int sock, void *buf, size_t max_len, int flags,
			       struct sockaddr *src_addr, socklen_t *addrlen)
{
	return zsock_recvfrom(sock, buf, max_len, flags, src_addr, addrlen);
}

static inline int poll(struct zsock_pollfd *fds, int nfds, int timeout)
{
	return zsock_poll(fds, nfds, timeout);
}

static inline int getsockopt(int sock, int level, int optname,
			     void *optval, socklen_t *optlen)
{
	return zsock_getsockopt(sock, level, optname, optval, optlen);
}

static inline int setsockopt(int sock, int level, int optname,
			     const void *optval, socklen_t optlen)
{
	return zsock_setsockopt(sock, level, optname, optval, optlen);
}

static inline int getaddrinfo(const char *host, const char *service,
			      const struct zsock_addrinfo *hints,
			      struct zsock_addrinfo **res)
{
	return zsock_getaddrinfo(host, service, hints, res);
}

static inline void freeaddrinfo(struct zsock_addrinfo *ai)
{
	zsock_freeaddrinfo(ai);
}

static inline const char *gai_strerror(int errcode)
{
	return zsock_gai_strerror(errcode);
}

static inline int getnameinfo(const struct sockaddr *addr, socklen_t addrlen,
			      char *host, socklen_t hostlen,
			      char *serv, socklen_t servlen, int flags)
{
	return zsock_getnameinfo(addr, addrlen, host, hostlen,
				 serv, servlen, flags);
}

#define addrinfo zsock_addrinfo

static inline int gethostname(char *buf, size_t len)
{
	return zsock_gethostname(buf, len);
}

static inline int inet_pton(sa_family_t family, const char *src, void *dst)
{
	return zsock_inet_pton(family, src, dst);
}

#else

struct addrinfo {
	int ai_flags;
	int ai_family;
	int ai_socktype;
	int ai_protocol;
	socklen_t ai_addrlen;
	struct sockaddr *ai_addr;
	char *ai_canonname;
	struct addrinfo *ai_next;
};

#include <net/socket_offload.h>

static inline int inet_pton(sa_family_t family, const char *src, void *dst)
{
	if ((family != AF_INET) && (family != AF_INET6)) {
		errno = EAFNOSUPPORT;
		return -1;
	}

	if (net_addr_pton(family, src, dst) == 0) {
		return 1;
	} else {
		return 0;
	}
}

#endif /* !defined(CONFIG_NET_SOCKETS_OFFLOAD) */

#define POLLIN ZSOCK_POLLIN
#define POLLOUT ZSOCK_POLLOUT
#define POLLERR ZSOCK_POLLERR
#define POLLHUP ZSOCK_POLLHUP
#define POLLNVAL ZSOCK_POLLNVAL

#define MSG_PEEK ZSOCK_MSG_PEEK
#define MSG_DONTWAIT ZSOCK_MSG_DONTWAIT

#define SHUT_RD ZSOCK_SHUT_RD
#define SHUT_WR ZSOCK_SHUT_WR
#define SHUT_RDWR ZSOCK_SHUT_RDWR

static inline char *inet_ntop(sa_family_t family, const void *src, char *dst,
			      size_t size)
{
	return zsock_inet_ntop(family, src, dst, size);
}

#define EAI_BADFLAGS DNS_EAI_BADFLAGS
#define EAI_NONAME DNS_EAI_NONAME
#define EAI_AGAIN DNS_EAI_AGAIN
#define EAI_FAIL DNS_EAI_FAIL
#define EAI_NODATA DNS_EAI_NODATA
#define EAI_MEMORY DNS_EAI_MEMORY
#define EAI_SYSTEM DNS_EAI_SYSTEM
#define EAI_SERVICE DNS_EAI_SERVICE
#endif /* defined(CONFIG_NET_SOCKETS_POSIX_NAMES) */

#define SOL_SOCKET 1

/* Socket options for SOL_SOCKET level */
#define SO_REUSEADDR 2
#define SO_ERROR 4
#define SO_RCVTIMEO 20
#define SO_BINDTODEVICE 25

/* Socket options for IPPROTO_TCP level */
#define TCP_NODELAY 1

/* Socket options for IPPROTO_IPV6 level */
#define IPV6_V6ONLY 26

/* Interface description structure */
#define IFNAMSIZ 64

struct ifreq {
    char ifr_name[IFNAMSIZ]; /* Interface name */
};

/* Protocol level for PDN. */
#define SOL_PDN 514

/* Socket options for SOL_PDN level */
#define SO_PDN_AF 1

/* Protocol level for DFU. */
#define SOL_DFU 515

/* Socket options for SOL_DFU level */
#define SO_DFU_FW_VERSION 1
#define SO_DFU_RESOURCE 2
#define SO_DFU_TIMEO 3
#define SO_DFU_APPLY 4
#define SO_DFU_REVERT 5
#define SO_DFU_BACKUP_DELETE 6
#define SO_DFU_OFFSET 7

#ifdef __cplusplus
}
#endif

#include <syscalls/socket.h>

/**
 * @}
 */

#endif /* ZEPHYR_INCLUDE_NET_SOCKET_H_ */
