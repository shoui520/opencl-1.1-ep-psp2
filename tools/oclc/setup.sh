#!/bin/sh
# Build the host OpenCL C compiler for the Vita: build/oclc/oclc.
#
#   tools/oclc/setup.sh
#
# oclc drives IMG's OpenCL front end (liboclcompiler.so from the DDK 1.12
# SGX544 userspace for ARMv7 musl) under qemu-arm, with PVR_PSP2's USC
# (patched with patches/pvr_psp2) in place of IMG's libusc.so.  Nothing of
# IMG's is kept in this repository: the front end is downloaded from
# Allwinner's Tina Linux package repository, and an Alpine Linux armhf
# sysroot is downloaded to run it, all checked against the hashes below.
#
# Needs Linux, qemu-arm (user mode, e.g. the qemu-user package), clang with
# the ARM target, lld, curl, git, tar and sha256sum.  Everything goes into
# build/oclc; the PVR_PSP2 submodule is left as it is.  Then:
#
#   OCLC=build/oclc/oclc python3 tools/oclc/clbin.py -o vadd.bin vadd.cl
set -eu

die() {
	echo "setup.sh: error: $*" >&2
	exit 1
}

# --- host checks, before anything is downloaded or built ---

[ "$(uname -s)" = Linux ] || die "this needs Linux (found $(uname -s)): IMG's compiler is an ARM Linux binary run through qemu-arm's user mode, which only exists on Linux. On Windows use WSL2."

QEMU=$(command -v qemu-arm || command -v qemu-arm-static || true)
[ -n "$QEMU" ] || die "qemu-arm not found. Install QEMU's user mode emulation: apt install qemu-user (Debian/Ubuntu), dnf install qemu-user (Fedora), pacman -S qemu-user (Arch)."
"$QEMU" --version >/dev/null 2>&1 || die "$QEMU doesn't run"

for tool in clang ld.lld curl git tar sha256sum sed; do
	command -v $tool >/dev/null 2>&1 || die "$tool not found"
done
clang -print-targets 2>/dev/null | grep -qw arm || die "this clang has no ARM target"
CLANG_INC=$(clang -print-resource-dir)/include
[ -f "$CLANG_INC/stddef.h" ] || die "clang's own headers aren't in $CLANG_INC"

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
P=$ROOT/third_party/PVR_PSP2
[ -f "$P/tools/intern/usc2/usc.c" ] || die "third_party/PVR_PSP2 is empty: run git submodule update --init"

O=$ROOT/build/oclc
DL=$O/dl
A=$O/sysroot
mkdir -p "$DL"

# --- downloads ---

TINA=https://raw.githubusercontent.com/tinalinux/package/59a58ef749ad0d59d3d33fe7e9bf17503819c776/libs/sgx544-um/fbdev/lib
ALPINE=https://dl-cdn.alpinelinux.org/alpine/v3.20

fetch() {	# url file sha256
	if [ -f "$DL/$2" ] && echo "$3  $DL/$2" | sha256sum -c --status; then
		return
	fi
	echo "downloading $1"
	curl -fL --retry 3 -o "$DL/$2.part" "$1" || die "couldn't download $1"
	echo "$3  $DL/$2.part" | sha256sum -c --status ||
		die "$2 doesn't match its sha256 (the source changed?): $1"
	mv "$DL/$2.part" "$DL/$2"
}

fetch $TINA/liboclcompiler.so.1.12.3670983 liboclcompiler.so \
	e668dc4de37b70fb3c4a19ba98a15a6a52b7720b4f6a63d914fae63295434e5a
fetch $TINA/libusc.so.1.12.3670983 libusc_img.so \
	a52d3e1525d574141b045588ad1c3c021945b1697036339027eec61fc554eafd
fetch $ALPINE/releases/armhf/alpine-minirootfs-3.20.3-armhf.tar.gz minirootfs.tar.gz \
	39e396324dc765beb4eef6b1e4d52de3143101094acbe48b837623e4e6bae500
fetch $ALPINE/main/armhf/musl-dev-1.2.5-r4.apk musl-dev.apk \
	9cedc1293408dfec026238f8d6d1e6a134cc6edd4c6577604ec17b713c650cd6
fetch $ALPINE/main/armhf/libgcc-13.2.1_git20240309-r1.apk libgcc.apk \
	9ed474be6d9c3e016bdc4ae509691b99d45a11d3f7832c05ffa7e7707bb9f7e4
fetch "$ALPINE/main/armhf/libstdc++-13.2.1_git20240309-r1.apk" libstdc++.apk \
	c915a11c9644615410ad22ec15563d5a1ba4ffe17d6cc7eaf5415b65832e164e

# --- the armhf sysroot ---

rm -rf "$A"
mkdir -p "$A/usr/lib/pvr"
tar -xzf "$DL/minirootfs.tar.gz" -C "$A" 2>/dev/null || die "couldn't unpack the Alpine minirootfs"
for apk in musl-dev libgcc libstdc++; do
	tar -xzf "$DL/$apk.apk" -C "$A" --exclude='.PKGINFO' --exclude='.SIGN.*' 2>/dev/null ||
		die "couldn't unpack $apk.apk"
done
cp "$DL/liboclcompiler.so" "$DL/libusc_img.so" "$A/usr/lib/pvr/"

# --- PVR_PSP2's USC, USP and assembler with patches/pvr_psp2 applied ---

S=$O/pvr
rm -rf "$S"
mkdir -p "$S"
git -C "$P" archive HEAD tools/intern | tar -x -C "$S" || die "couldn't export PVR_PSP2's tools/intern"
# its own repository, so git apply doesn't resolve paths against this one
git -C "$S" init -q
for patch in "$ROOT"/patches/pvr_psp2/*.patch; do
	git -C "$S" apply --whitespace=nowarn "$patch" ||
		die "$(basename "$patch") doesn't apply to PVR_PSP2 $(git -C "$P" rev-parse --short HEAD)"
done

CC="clang --target=armv7-alpine-linux-musleabihf -mfloat-abi=hard -nostdinc -isystem $CLANG_INC
    -isystem $A/usr/include --sysroot=$A -fuse-ld=lld -nostdlib -L$A/usr/lib -L$A/lib
    -Wl,--dynamic-linker=/lib/ld-musl-armhf.so.1"
INC="-I$P/intermediates/sgxsupport -I$P/intermediates/errata -I$P/include/gpu_es4
     -I$P/include/gpu_es4/eurasia/include4 -I$P/include/gpu_es4/eurasia/hwdefs
     -I$P/include/gpu_es4/eurasia/services4/include -I$S/tools/intern/usc2 -I$S/tools/intern/usp
     -I$S/tools/intern/useasm -I$P/eurasiacon/include"
DEFS="-include stdlib.h -include psp2_pvr_desc.h -DIMG_ABORT=abort -DLINUX -DDEBUG -DGEN_HW_CODE
      -DINCLUDE_SGX_FEATURE_TABLE -DINCLUDE_SGX_BUG_TABLE -DOUTPUT_USPBIN -DUSER -DSUPPORT_SGX
      -DSUPPORT_SGX543 -DSUPPORT_SGX544"
REN="-DPVRUniFlexCreateContext=psp2_PVRUniFlexCreateContext
     -DPVRUniFlexCompileToUspBin=psp2_PVRUniFlexCompileToUspBin
     -DPVRUniFlexDestroyUspBin=psp2_PVRUniFlexDestroyUspBin
     -DPVRUniFlexDestroyContext=psp2_PVRUniFlexDestroyContext"
LIBS="$A/usr/lib/libc.so $A/usr/lib/libgcc_s.so.1 $A/usr/lib/crtn.o"

USC="asm cdg cfa data dce debug dgraph domcalc dualissue efo execpred f16opt finalise groupinst
     hw icvt_c10 icvt_core icvt_f16 icvt_f16_vec icvt_f32 icvt_f32_vec icvt_i32 icvt_mem indexreg
     inst_usc intcvt iregalloc iselect layout pconvert precovr pregalloc regalloc reggroup regpack
     reorder ssa usc usc_utils usedef uspbin vec34"
USEASM="specialregs specialregs_vec useasm usedisasm useopt usetab utils"
USP="usp_finalise hwinst usp uspshader usp_inputdata usp_instblock usp_resultref usp_sample
     usp_texwrite"

echo "compiling PVR_PSP2's USC and USP for armhf"
mkdir -p "$O/obj" "$O/arm"
OBJS=""
pids=""
for dir_files in "usc2:$USC" "useasm:$USEASM" "usp:$USP"; do
	dir=${dir_files%%:*}
	for f in ${dir_files#*:}; do
		o=$O/obj/${dir}_$f.o
		OBJS="$OBJS $o"
		$CC -c -O1 -g -fPIC -w -fno-strict-aliasing $INC $DEFS $REN \
			"$S/tools/intern/$dir/$f.c" -o "$o" &
		pids="$pids $!"
	done
done
failed=0
for pid in $pids; do
	wait $pid || failed=1
done
[ $failed = 0 ] || die "compiling PVR_PSP2 failed (see above)"

echo "linking libusc.so and oclc"
$CC -c -O1 -g -fPIC -w $INC $DEFS "$ROOT/tools/oclc/usc_shim.c" -o "$O/obj/shim.o" ||
	die "compiling usc_shim.c failed"
$CC -shared -Wl,-soname,libusc.so -Wl,--version-script="$ROOT/tools/oclc/usc_shim.map" \
	"$A/usr/lib/crti.o" "$O/obj/shim.o" $OBJS $LIBS -o "$O/arm/libusc.so" ||
	die "linking libusc.so failed"
sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/\t"/' -e 's/$/\\n"/' \
	"$ROOT/tools/oclc/prelude.cl" > "$O/obj/prelude.inc"
$CC -O1 -g -w -I"$O/obj" "$ROOT/tools/oclc/oclc.c" "$A/usr/lib/crt1.o" "$A/usr/lib/crti.o" $LIBS \
	-o "$O/arm/oclc" || die "building oclc failed"

cat > "$O/oclc" <<'EOF'
#!/bin/sh
# Run oclc (tools/oclc/oclc.c) under qemu-arm; written by tools/oclc/setup.sh.
D=$(cd "$(dirname "$0")" && pwd)
QEMU=$(command -v qemu-arm || command -v qemu-arm-static) || {
	echo "oclc: qemu-arm not found" >&2
	exit 1
}
QEMU_LD_PREFIX=$D/sysroot LD_LIBRARY_PATH=$D/arm:/usr/lib/pvr exec "$QEMU" -L "$D/sysroot" "$D/arm/oclc" "$@"
EOF
chmod +x "$O/oclc"

# --- check it compiles a kernel ---

T=$O/selftest
rm -rf "$T"
mkdir -p "$T"
cat > "$T/vadd.cl" <<'EOF'
__kernel void vadd(__global const int *a, __global const int *b, __global int *c)
{
	size_t i = get_global_id(0);
	c[i] = a[i] + b[i];
}
EOF
OCLC_GXM=1 "$O/oclc" -m "$T/vadd.meta" "$T/vadd.cl" > "$T/log" 2>&1 ||
	die "oclc failed on a test kernel (log: $T/log)"
grep -q '^kernel vadd ' "$T/vadd.meta" || die "oclc didn't compile the test kernel (log: $T/log)"

echo "built $O/oclc"
