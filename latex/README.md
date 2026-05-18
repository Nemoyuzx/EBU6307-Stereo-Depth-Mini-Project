# LaTeX Report Workspace

这个目录现在是报告的 LaTeX 编写入口。

## 文件说明

- `report_en.tex`：英文版报告源文件
- `report_zh.tex`：中文版报告源文件
- `build.sh`：编译两份报告并同步更新根目录提交 PDF

## 编译方式

在项目根目录执行：

```bash
bash latex/build.sh
```

或进入 `latex/` 目录执行：

```bash
cd latex
bash build.sh
```

## 输出结果

- `latex/report_en.pdf`
- `latex/report_zh.pdf`
- `../EBU6307_ZIXI_YU_231223210.pdf`
- `../EBU6307_ZIXI_YU_231223210_zh.pdf`

## 维护说明

- 以后优先编辑本目录下的 `.tex` 文件。
- 根目录下的 Markdown 报告保留为历史参考，不再作为主编写入口。