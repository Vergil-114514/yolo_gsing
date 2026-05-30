# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## 5. 测试文件清理

**用完即删。测试文件不是永久代码。**

- 为验证功能临时创建的测试文件（脚本、临时页面、mock 数据等），验证完成后立即删除。
- 不要留下 `test-foo.js`、`debug-bar.ts` 之类的临时文件污染项目。
- 如果测试逻辑有长期价值，整合进正式测试套件，不要单独遗留在项目里。

## 6. 项目说明与变更日志

**每次改动都记录在 项目说明.md 中。**

- 项目根目录维护一个 `项目说明.md`，包含项目概述、技术栈、目录结构、运行方式等。
- 每次对项目的改动（新功能、bug 修复、重构、配置变更等）都要在文件末尾的变更日志区域追加一条记录。
- 日志格式：`YYYY-MM-DD | 简要描述改动内容 | 改动原因`
- 保持文件简洁，新成员应该能通过它快速上手项目。

---

## 7. 代码注释规范

**按规范写注释，不偷懒。**

- 所有公开函数、类、方法必须有注释说明用途、参数、返回值。
- 注释解释 **WHY**（为什么这样做），不解释 **WHAT**（代码做了什么）——代码本身已经说了 WHAT。
- 复杂逻辑、非常规写法、hack/workaround 必须注释说明原因。
- 注释使用项目统一的语言和风格（中文项目用中文，英文项目用英文）。
- 不写废话注释（如 `// 给 x 加 1` 对应 `x += 1`）。
- 不要用注释掩盖烂代码——如果代码需要大段注释才能理解，先考虑重构代码本身。

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
