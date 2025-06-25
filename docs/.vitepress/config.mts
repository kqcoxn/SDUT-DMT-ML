import { defineConfig } from "vitepress";

import nav from "./configs/nav";
import sidebar from "./configs/sidebar";

export default defineConfig({
  title: "Machine Learning",
  titleTemplate: "ML Documentation of SDUT-DMT",
  description: "机器学习课程在线文档",

  lang: "zh-CN",
  base: "/ml/",
  head: [["link", { rel: "icon", href: "/ml/logo.png" }]],

  themeConfig: {
    logo: "/logo.png",
    outlineTitle: "本页大纲",
    outline: [2, 3],

    nav: nav,
    sidebar,

    search: {
      provider: "local",
    },
    socialLinks: [
      { icon: "github", link: "https://github.com/kqcoxn/SDUT-DMT-ML" },
    ],
  },

  markdown: {
    math: true,
  },
});
