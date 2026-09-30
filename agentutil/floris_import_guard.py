# -*- coding: utf-8 -*-
"""FLORIS 导入守卫：挡住"用错 Python 解释器 / 基础 conda 环境污染 sys.path"导致的
`pyexpat` DLL 加载失败。

症状
----
    ImportError: DLL load failed while importing pyexpat: 操作系统无法运行 %1。
    （栈：floris → cut_plane → matplotlib.pyplot → ... → plistlib → pyexpat）

根因（本机实测，2026）
----------------------
**主因是解释器不匹配**，不是路径污染（路径污染只是次要触发条件）：

    F:\\Anaconda\\python.exe            → Python 3.12.7    （基础环境）
    F:\\anaconda\\envs\\wsis\\python.exe → Python 3.12.14   （本项目环境）

若用**基础环境**的 `python.exe` 去加载 `wsis` 的 `site-packages`，则
`pyexpat.pyd`（按 3.12.14 编译）在 3.12.7 下无法加载 ⇒ 上面那条报错。
**3.12.x 之间也有 ABI 差异，所以"都是 3.12"并不够。**

做法
----
1. `check_interpreter()` —— 在主流程开头调用：**立刻**判断"当前解释器是不是本环境"，
   不是就打印可直接照抄的命令并退出（而不是让用户看到一条 DLL 报错猜半天）。
2. `sanitize()` —— 在任何 `import floris` **之前**调用：把"其它 Python 安装"的路径
   从 `sys.path` 里剔除（次要防线；`sys.path[0]` 与空串保留，否则 `agentutil`
   内部互相 import 会断）。
3. **换机器时的退路**（2026-09-30 新增，本机实测触发）：
   * `EXPECTED_PYTHON` 可用环境变量 **`FLORIS_PYTHON`** 覆盖 ⇒ 换环境不必改源码；
   * 期望解释器路径在本机**不存在**时（例如仓库被复制到没有 `F:` 盘的机器），
     路径判据本身失效 ⇒ **自动退回功能判据**：直接 `import pyexpat`，
     成功即放行（这才是本守卫真正要挡的东西），失败才报错退出。
     退回时会在报告中写明 `fallback`，不会静默改变判据。
   ⚠ 路径**存在但不等**（同一台机器上用错解释器）时行为不变：仍然硬失败。

用法（放在所有第三方 import 之前）
----------------------------------
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from floris_import_guard import check_interpreter, sanitize
    check_interpreter()          # 解释器错了就在这里退出（给出命令）
    sanitize(verbose=False)      # 顺手清掉污染路径
    # ... 之后才可以 import floris / yawflor
"""
import os
import sys

# 本项目的官方解释器（agentutil/README.md 记载）。
#   * 可用环境变量 FLORIS_PYTHON 覆盖（换机器/换虚拟环境时无需改源码）；
#   * 该路径在本机不存在时 ⇒ 路径判据不可用，自动退回"功能判据"（见 check_interpreter）。
_DEFAULT_PYTHON = r'F:\anaconda\envs\wsis\python.exe'
EXPECTED_PYTHON = os.environ.get('FLORIS_PYTHON') or _DEFAULT_PYTHON
_REMOVED = []


def ensure_printable_streams():
    """把 `stdout`/`stderr` 切到 UTF-8 + `errors='replace'`（幂等，可重复调用）。

    ⚠ 为什么必须放在**库里**而不是只放在 `__main__` 里
    ---------------------------------------------------
    本守卫的报告含 `⇒`(U+21D2) / `⚠`(U+26A0) 等**非 GBK** 字符，而它是在**被 import
    的那一刻**就被调用的（`agentutil/probe_engine_random_actions.py:53` 的 `_chk()`，
    而该模块又是 `aifflor_turbine.build_model()` 的依赖）。此时进程通常不是 `__main__`，
    没人替我们 reconfigure ⇒ **stdout 一旦不是真控制台**（被管道接走、
    `> log.txt`、被父进程捕获）就退化成 cp936 严格编码，打印 `⇒` 直接抛
    `UnicodeEncodeError` —— 守卫本身变成新的故障源（本机实测）。

    真控制台下 Python 走 `WriteConsoleW`（encoding 本就是 utf-8），本函数等价于空操作。
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError, OSError):
            pass


def check_interpreter(expected=EXPECTED_PYTHON, fatal=True, verbose=False):
    """判断当前解释器是不是本项目的环境；不是则给出可直接照抄的命令。

    判据两档（顺序固定）：
      ① **路径判据**：`sys.executable` 与 `expected` 逐字相同（期望路径存在时用这档）；
      ② **功能判据**：期望路径在本机不存在（换机器/仓库被复制）时退回 —— 只验
         `import pyexpat` 能否成功。3.12.x 之间的 ABI 不匹配正是它失败的原因，
         所以这档仍能挡住本守卫真正要挡的故障，只是不再要求路径逐字相同。

    Returns
    -------
    dict(ok, reason, executable, version, expected, fallback, has_floris)
        fallback=True ⇒ 本机没有期望解释器，结论来自功能判据 ②（非路径判据）。
    """
    ensure_printable_streams()          # 先修流，再打印（否则报告里的 ⇒/⚠ 会崩）
    exe = os.path.normcase(os.path.abspath(sys.executable))
    exp = os.path.normcase(os.path.abspath(expected)) if expected else None
    ver = sys.version.split()[0]
    ok = (exp is None) or (exe == exp)
    fallback = False
    reason = None
    if not ok:
        expected_missing = (exp is not None) and (not os.path.exists(expected))
        if expected_missing:
            fallback = True
            try:
                import pyexpat  # noqa: F401
                ok = True
            except Exception as e:                      # pragma: no cover
                reason = ('期望解释器 %s 在本机不存在，退回功能判据后仍失败：\n'
                          '    当前 : %s  (Python %s)\n'
                          '    %s: %s\n'
                          '  请用 FLORIS_PYTHON=<本项目解释器> 显式指定，或清掉 '
                          'PYTHONHOME/PYTHONPATH。'
                          % (expected, sys.executable, ver, type(e).__name__, e))
        else:
            reason = ('当前解释器不是本项目环境。\n'
                      '    当前 : %s  (Python %s)\n'
                      '    应为 : %s\n'
                      '  3.12.x 之间的扩展模块（pyexpat.pyd 等）**ABI 不兼容**，用错解释器会报\n'
                      '  "ImportError: DLL load failed while importing pyexpat"。' %
                      (sys.executable, ver, expected))
    if ok and not fallback:
        try:
            import pyexpat  # noqa: F401
        except Exception as e:                      # pragma: no cover
            ok, reason = False, ('解释器正确但 pyexpat 仍加载失败：%s: %s\n'
                                 '  多半是 PYTHONHOME/PYTHONPATH 指向了别的安装。'
                                 % (type(e).__name__, e))
    info = dict(ok=ok, reason=reason, executable=sys.executable, version=ver,
                expected=expected, fallback=fallback)
    if verbose or not ok or fallback:
        report(info)
    if not ok and fatal:
        print('\n' + '=' * 80)
        print('请在**普通终端**里改成下面这条命令重跑（把 <脚本> 换成你要跑的脚本）:')
        print('    cd "%s"' % os.getcwd())
        print('    "%s" agentutil\\<脚本>.py' % expected)
        print('\n若已经在用正确的解释器，请先清掉这两个环境变量再试:')
        print('    set PYTHONPATH=')
        print('    set PYTHONHOME=')
        print('  也可以直接用 FLORIS_PYTHON=<你的解释器> 指定本项目环境。')
        print('=' * 80)
        raise SystemExit(3)
    return info


def _prefixes():
    """本解释器合法的路径前缀（全部小写、去尾分隔符）。"""
    out = set()
    for p in (sys.prefix, sys.exec_prefix, getattr(sys, 'base_prefix', sys.prefix)):
        if p:
            out.add(os.path.normcase(os.path.normpath(p)))
    return out


def _is_other_python(path):
    """判断 path 是否像"另一个 Python 安装的根 / 其 Lib"。

    特征：目录里同时存在 `Lib` 与 `DLLs`（Windows 布局），或存在 `python*.dll`。
    只做**保守**判定：宁可不删，也不误删本环境自己的路径。
    """
    try:
        if not os.path.isdir(path):
            return False
    except OSError:
        return False
    has_lib = os.path.isdir(os.path.join(path, 'Lib'))
    has_dlls = os.path.isdir(os.path.join(path, 'DLLs'))
    return bool(has_lib and has_dlls)


def sanitize(verbose=False):
    """剔除污染路径。返回被剔除的路径列表。幂等，可重复调用。"""
    good = _prefixes()
    keep, removed = [], []
    for i, p in enumerate(sys.path):
        if i == 0:                                   # 脚本目录 / '' 必须保留
            keep.append(p)
            continue
        try:
            np_ = os.path.normcase(os.path.normpath(p)) if p else p
        except OSError:
            keep.append(p)
            continue
        if not p:
            keep.append(p)
            continue
        # 本环境前缀下的路径一律保留
        if any(np_ == g or np_.startswith(g + os.sep) for g in good):
            keep.append(p)
            continue
        # 其它 Python 安装的根 或 其 Lib/DLLs ⇒ 剔除
        base = os.path.dirname(np_) if os.path.basename(np_).lower() in ('lib', 'dlls',
                                                                        'libs') else np_
        if _is_other_python(base) or _is_other_python(np_):
            removed.append(p)
            continue
        keep.append(p)

    sys.path[:] = keep
    _REMOVED.extend(removed)
    if verbose:
        report()
    return removed


def report(info=None):
    """打印诊断信息（出问题时让用户一眼看出是解释器问题还是路径污染）。"""
    ensure_printable_streams()
    if info is None:
        info = check_interpreter(fatal=False)
    if info.get('fallback'):
        # 紧凑报告：本机没有期望解释器 ⇒ 结论来自功能判据（pyexpat），不刷整块诊断
        print('[floris_import_guard] 期望解释器 : %s（**本机不存在**）'
              % info.get('expected'))
        print('[floris_import_guard] 当前解释器 : %s  (Python %s)'
              % (info['executable'], info['version']))
        print('[floris_import_guard] ⇒ 退回功能判据: pyexpat 加载%s ⇒ %s'
              '（需要精确锁定环境时用 FLORIS_PYTHON=<解释器>）'
              % ('成功' if info['ok'] else '失败', '放行' if info['ok'] else '中止'))
        if info.get('reason'):
            print(info['reason'])
        return
    print('[floris_import_guard] 期望解释器  : %s' % (info.get('expected') or EXPECTED_PYTHON))
    print('[floris_import_guard] 当前解释器  : %s' % sys.executable)
    print('[floris_import_guard] Python 版本 : %s' % sys.version.split()[0])
    print('[floris_import_guard] sys.prefix  : %s' % sys.prefix)
    print('[floris_import_guard] PYTHONPATH  : %r' % os.environ.get('PYTHONPATH'))
    print('[floris_import_guard] PYTHONHOME  : %r' % os.environ.get('PYTHONHOME'))
    print('[floris_import_guard] MPLCONFIGDIR: %r' % os.environ.get('MPLCONFIGDIR'))
    if info is None:
        info = check_interpreter(fatal=False)
    print('[floris_import_guard] 解释器正确? : %s' % ('是' if info['ok'] else '**否**'))
    print('[floris_import_guard] 已剔除的污染路径 (%d):' % len(_REMOVED))
    for p in _REMOVED:
        print('    - %s' % p)
    if not _REMOVED:
        print('    （无 —— sys.path 干净）')
    try:
        import pyexpat
        print('[floris_import_guard] pyexpat OK : %s' % pyexpat.__file__)
    except Exception as e:
        print('[floris_import_guard] ⚠ pyexpat 失败: %s: %s' % (type(e).__name__, e))


if __name__ == '__main__':
    ensure_printable_streams()
    _i = check_interpreter(fatal=False)
    sanitize()
    report(_i)
    print('\n自检结果: %s' % ('通过 ✅' if _i['ok'] else '**解释器不对** ❌（见上面提示）'))
    sys.exit(0 if _i['ok'] else 3)
