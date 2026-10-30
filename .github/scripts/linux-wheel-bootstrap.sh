#!/usr/bin/env bash
# manylinux2014 keeps glibc 2.17; its default GCC 10 lacks floating to_chars.
set -euo pipefail
cuda_version=${1:?expected cpu, 11.8, or 12.1}
case "$cuda_version" in
  cpu) installer=; cuda_url= ;;
  11.8) installer=cuda_11.8.0_520.61.05_linux.run
    cuda_url=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/$installer ;;
  12.1) installer=cuda_12.1.1_530.30.02_linux.run
    cuda_url=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/$installer ;;
  *) echo "Unsupported CUDA selector: $cuda_version" >&2; exit 2 ;;
esac
# CentOS 7 mirrorlists are retired. Keep signature and repository checksum checks.
for repo in /etc/yum.repos.d/CentOS-*.repo; do
  [ ! -f "$repo" ] || mv "$repo" "$repo.disabled"
done
cat >/etc/yum.repos.d/wheel-vault.repo <<'REPOS'
[wheel-base]
name=CentOS 7.9.2009 base archive
baseurl=https://vault.centos.org/7.9.2009/os/x86_64/
enabled=1
gpgcheck=1
gpgkey=https://www.centos.org/keys/RPM-GPG-KEY-CentOS-7
[wheel-updates]
name=CentOS 7.9.2009 updates archive
baseurl=https://vault.centos.org/7.9.2009/updates/x86_64/
enabled=1
gpgcheck=1
gpgkey=https://www.centos.org/keys/RPM-GPG-KEY-CentOS-7
[wheel-sclo-rh]
name=CentOS 7.9.2009 SCLo RH archive
baseurl=https://vault.centos.org/7.9.2009/sclo/x86_64/rh/
enabled=1
gpgcheck=1
gpgkey=https://www.centos.org/keys/RPM-GPG-KEY-CentOS-SIG-SCLo
REPOS
yum install -y devtoolset-11-runtime-11.0-3.el7 \
  devtoolset-11-gcc-11.2.1-9.1.el7 devtoolset-11-gcc-c++-11.2.1-9.1.el7 \
  devtoolset-11-libstdc++-devel-11.2.1-9.1.el7
export PATH=/opt/rh/devtoolset-11/root/usr/bin:$PATH
export CC=/opt/rh/devtoolset-11/root/usr/bin/gcc
export CXX=/opt/rh/devtoolset-11/root/usr/bin/g++
[ "$($CXX -dumpversion | cut -d. -f1)" = 11 ]
probe_tmp=$(mktemp -d)
cat >"$probe_tmp/to_chars.cpp" <<'CPP'
#include <charconv>
#include <system_error>
int main() { char out[64]; auto r = std::to_chars(out, out + sizeof(out), 1.25);
  if (r.ec != std::errc{} || r.ptr - out != 4 || out[0] != '1' || out[3] != '5') return 1;
  auto f = std::to_chars(out, out + sizeof(out), 1.25f, std::chars_format::scientific);
  if (f.ec != std::errc{}) return 2;
  auto d = std::to_chars(out, out + sizeof(out), 1.25, std::chars_format::scientific);
  return d.ec == std::errc{} ? 0 : 3; }
CPP
"$CXX" -std=c++17 "$probe_tmp/to_chars.cpp" -o "$probe_tmp/to_chars"
"$probe_tmp/to_chars"
rm -rf "$probe_tmp"
# The archive's Boost 1.53 is too old for CMake's >=1.70 contract.
# Retain the prior 1.82 source build, using a private scratch directory.
boost_tmp=$(mktemp -d)
trap 'rm -rf "$boost_tmp"' EXIT
curl --fail --location --retry 3 \
  https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz \
  -o "$boost_tmp/boost.tar.gz"
echo "66a469b6e608a51f8347236f4912e27dc5c60c60d7d53ae9bfe4683316c6f04c  $boost_tmp/boost.tar.gz" > "$boost_tmp/checksum"
# Published digest: https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz.json
sha256sum -c "$boost_tmp/checksum"
tar --no-same-owner -xzf "$boost_tmp/boost.tar.gz" -C "$boost_tmp"
(cd "$boost_tmp/boost_1_82_0" &&
 ./bootstrap.sh --prefix=/usr/local --with-libraries=regex,system,filesystem &&
 ./b2 install --prefix=/usr/local threading=multi link=shared variant=release -j2)
if [ "$cuda_version" != cpu ]; then
  if [ ! -x /usr/local/cuda/bin/nvcc ]; then
    curl --fail --location --retry 3 "$cuda_url" -o "$boost_tmp/$installer"
    sh "$boost_tmp/$installer" --silent --toolkit --toolkitpath=/usr/local/cuda
  fi
  /usr/local/cuda/bin/nvcc --version | grep -F "release $cuda_version,"
fi
