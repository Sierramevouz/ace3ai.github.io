# ACE3 英文网站部署指南

## 项目概述

ACE3英文网站是一个完整的多页面网站，包含以下页面：

- 首页 (index.html)
- 价格页面 (pricing.html)
- 文档页面 (docs.html)
- 博客页面 (blog.html)
- 联系我们页面 (contact.html)
- 关于我们页面 (about.html)
- 登录页面 (login.html)

所有页面均保持统一的设计风格，并包含轨迹流动动画效果。

## 文件结构

```
ace3_website_complete/
├── css/
│   ├── responsive.css    # 响应式设计样式
│   └── styles.css        # 主要样式文件
├── images/
│   └── logo.svg          # ACE3 logo
├── js/
│   ├── main.js           # 主要JavaScript功能
│   └── trajectory-flow.js # 轨迹流动动画效果
├── index.html            # 首页
├── pricing.html          # 价格页面
├── docs.html             # 文档页面
├── blog.html             # 博客页面
├── contact.html          # 联系我们页面
├── about.html            # 关于我们页面
└── login.html            # 登录页面
```

## 部署方法

### 方法1：使用静态网站托管服务

1. 解压`ace3_website_complete.tar.gz`压缩包
2. 将解压后的整个`ace3_website_complete`目录上传到您选择的静态网站托管服务，如GitHub Pages、Netlify、Vercel等
3. 按照托管服务的指南完成部署配置

### 方法2：使用传统Web服务器

1. 解压`ace3_website_complete.tar.gz`压缩包
2. 将解压后的整个`ace3_website_complete`目录上传到您的Web服务器
3. 配置Web服务器（如Apache、Nginx）指向该目录

### 方法3：本地测试

1. 解压`ace3_website_complete.tar.gz`压缩包
2. 在本地计算机上，进入解压后的`ace3_website_complete`目录
3. 使用Python启动一个简单的HTTP服务器：
   ```
   python -m http.server 8000
   ```
   或者使用Node.js：
   ```
   npx serve
   ```
4. 在浏览器中访问`http://localhost:8000`

## 自定义指南

### 修改Logo

如果您想使用自己的logo，请替换`images/logo.svg`文件。建议使用SVG格式以获得最佳显示效果。

### 修改颜色方案

网站的颜色方案定义在`css/styles.css`文件的`:root`部分。您可以修改这些变量来更改整个网站的颜色方案：

```css
:root {
    --primary-color: #6246ea;    /* 主要颜色 */
    --secondary-color: #38bdf8;  /* 次要颜色 */
    --tertiary-color: #0ea5e9;   /* 第三颜色 */
    --light-color: #e0f2fe;      /* 浅色 */
    --dark-color: #1e293b;       /* 深色 */
    --background-color: #ffffff; /* 背景色 */
    --text-color: #333333;       /* 文本颜色 */
    --light-purple: #e9e3ff;     /* 浅紫色 */
    --purple: #d8d0ff;           /* 紫色 */
    --blue: #c2e7ff;             /* 蓝色 */
    --light-blue: #e0f2fe;       /* 浅蓝色 */
}
```

### 调整轨迹流动效果

轨迹流动效果的参数可以在`js/trajectory-flow.js`文件中调整：

- 修改点的数量：更改`TrajectoryFlowManager`构造函数中的参数（默认为40）
- 修改点的大小：调整`TrajectoryPoint`构造函数中的`size`参数
- 修改点的颜色和透明度：调整`createPointElement`方法中的`backgroundColor`属性

### 添加新页面

如果您需要添加新页面，建议复制现有页面（如`about.html`）并修改内容，以保持整体风格一致。确保在新页面中包含以下元素：

1. 头部引用相同的CSS文件
2. 背景容器元素（用于轨迹流动效果）
3. 导航栏和页脚
4. 底部引用相同的JavaScript文件

## 注意事项

- 所有页面都已经过响应式设计测试，适配不同屏幕尺寸
- 轨迹流动效果在所有现代浏览器中都能正常工作
- 登录页面包含基本的表单验证功能，但不包含后端处理逻辑
- 联系页面中的地图是静态嵌入的，如需交互式地图，请替换为Google Maps或其他地图服务的嵌入代码

## 技术支持

如果您在部署或自定义过程中遇到任何问题，请随时联系我们获取技术支持。
