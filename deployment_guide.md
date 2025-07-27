# ACE3 英文网站部署指南

本文档提供了如何部署ACE3英文网站的详细说明。

## 文件结构

网站包含以下文件和目录：

```
ace3_website/
├── css/
│   ├── styles.css       # 主样式文件
│   └── responsive.css   # 响应式设计样式
├── js/
│   ├── trajectory-flow.js  # 轨迹流动效果实现
│   └── main.js          # 主要JavaScript功能
├── images/              # 图片目录（需要添加logo等图片）
└── index.html           # 主页HTML文件
```

## 部署步骤

### 方法1：使用静态网站托管服务

1. 将整个`ace3_website`目录上传到您选择的静态网站托管服务，如GitHub Pages、Netlify、Vercel等。
2. 按照托管服务的指南完成部署配置。

### 方法2：使用传统Web服务器

1. 将整个`ace3_website`目录上传到您的Web服务器。
2. 配置Web服务器（如Apache、Nginx）指向该目录。

### 方法3：本地测试

1. 在本地计算机上，进入`ace3_website`目录。
2. 使用Python启动一个简单的HTTP服务器：
   ```
   python -m http.server 8000
   ```
   或者使用Node.js：
   ```
   npx serve
   ```
3. 在浏览器中访问`http://localhost:8000`。

## 自定义说明

### 添加Logo

1. 将您的logo图片放入`images`目录。
2. 在`index.html`文件中更新logo图片路径：
   ```html
   <img src="images/your-logo.png" alt="ACE3 Logo">
   ```

### 修改颜色方案

如果需要调整颜色方案，请编辑`css/styles.css`文件中的以下部分：

```css
:root {
    --primary-color: #6246ea;    /* 主色调 */
    --secondary-color: #38bdf8;  /* 次要色调 */
    --tertiary-color: #0ea5e9;   /* 第三色调 */
    --light-color: #e0f2fe;      /* 浅色 */
    --dark-color: #1e293b;       /* 深色 */
    /* 其他颜色变量... */
}
```

### 调整轨迹流动效果

如果需要调整轨迹流动效果，请编辑`js/trajectory-flow.js`文件：

1. 修改轨迹点数量：
   ```javascript
   const trajectoryFlow = new TrajectoryFlowManager(40); // 将40改为您想要的数量
   ```

2. 调整轨迹点大小和速度：
   ```javascript
   const size = Math.random() * 10 + 5; // 5-15px
   const speed = Math.random() * 0.5 + 0.2; // 0.2-0.7
   ```

## 浏览器兼容性

网站已针对现代浏览器进行了优化，包括：
- Chrome（最新版本）
- Firefox（最新版本）
- Safari（最新版本）
- Edge（最新版本）

## 性能优化建议

为获得最佳性能：

1. 优化图片大小和格式（推荐使用WebP格式）。
2. 考虑使用CDN托管静态资源。
3. 如果网站内容增长，考虑将CSS和JavaScript文件进行压缩。

## 故障排除

如果轨迹流动效果不显示：
1. 检查浏览器控制台是否有JavaScript错误。
2. 确保`trajectory-flow.js`文件正确加载。
3. 验证`.background-container`和`.gradient-bg`元素存在于DOM中。

如有任何问题或需要进一步的帮助，请联系开发团队。
