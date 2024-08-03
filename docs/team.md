---
layout: page
---

<script setup>
import {
  VPTeamPage,
  VPTeamPageTitle,
  VPTeamMembers
} from 'vitepress/theme'

const members = [
  {
    avatar: '/ml/mjy.jpg',
    name: '马锦艺',
    title: '数媒2102 | 文档架构',
    links: [
      { icon: 'github', link: 'https://github.com/kqcoxn' },
    ]
  },
  {
    avatar: '/ml/shx.jpg',
    name: '宋欢修',
    title: '数媒2002老学长 | 伟大的资源提供者',
  },
]
</script>

<VPTeamPage>
  <VPTeamPageTitle>
    <template #title>
      文档贡献者
    </template>
    <template #lead>
      SDUT-DMT-ML Documentation Contributors
    </template>
  </VPTeamPageTitle>
  <VPTeamMembers size="medium"
    :members="members"
  />
</VPTeamPage>
