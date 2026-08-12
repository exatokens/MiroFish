"""
本体生成服务
接口1：分析文本内容，生成适合社会模拟的实体和关系类型定义
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# 本体生成的系统提示词
ONTOLOGY_SYSTEM_PROMPT = """You are an expert knowledge-graph ontology designer. Your task is to analyze the given text content and simulation requirement, then design entity types and relationship types suitable for a **social media simulation**.

**IMPORTANT: Output valid JSON only. Do not include any text outside the JSON.**

## Core Task Context

We are building a **social media simulation system** where:
- Every entity is an "account" or "actor" that can post, interact, and spread information on social media
- Entities influence each other through replies, shares, comments, and reactions
- We need to simulate how different stakeholders react and how information propagates

Therefore, **entities must be real-world actors capable of speaking and interacting on social media**:

**Valid entity types**:
- Specific individuals (public figures, key persons, opinion leaders, experts, ordinary users)
- Companies and corporations (including their official accounts)
- Organizations (universities, associations, NGOs, unions, etc.)
- Government bodies and regulators
- Media outlets (newspapers, TV stations, independent media, websites)
- Social media platforms themselves
- Representative groups (alumni associations, fan groups, advocacy groups, etc.)

**📈 For financial/stock analysis documents, also include**:
- Analysts (sell-side and buy-side research analysts)
- Institutional investors (hedge funds, mutual funds, pension funds)
- Retail investors (individual investors, forum users)
- Customers / clients (key enterprise customers of the company)
- Competitors (rival companies in the same industry)

**NOT valid** (do not use):
- Abstract concepts (e.g., "public opinion", "sentiment", "trends")
- Topics / themes (e.g., "academic integrity", "market volatility")
- Viewpoints / stances (e.g., "supporters", "opponents")

## Output Format

Output JSON with this exact structure:

```json
{
    "entity_types": [
        {
            "name": "EntityTypeName (English, PascalCase)",
            "description": "Short description (English, under 100 characters)",
            "attributes": [
                {
                    "name": "attribute_name (English, snake_case)",
                    "type": "text",
                    "description": "Attribute description"
                }
            ],
            "examples": ["Example entity 1", "Example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "RELATIONSHIP_NAME (English, UPPER_SNAKE_CASE)",
            "description": "Short description (English, under 100 characters)",
            "source_targets": [
                {"source": "SourceEntityType", "target": "TargetEntityType"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief summary of the text analysis (English)"
}
```

## Design Guidelines (CRITICAL)

### 1. Entity Type Design — Must Follow Strictly

**Count requirement: exactly 10 entity types**

**Hierarchy requirement (must include both specific and fallback types)**:

Your 10 entity types must follow this structure:

A. **Fallback types (required, placed last in the list)**:
   - `Person`: fallback for any individual not matching a more specific person type
   - `Organization`: fallback for any organization not matching a more specific organization type

B. **Specific types (8 types, designed based on the text content)**:
   - Design based on the key roles appearing in the text
   - Example for academic events: `Student`, `Professor`, `University`
   - Example for corporate/financial events: `Company`, `Executive`, `Analyst`, `InstitutionalInvestor`, `Customer`, `Competitor`, `Regulator`

**Why fallback types are needed**:
- Text often mentions minor characters ("an anonymous user", "a teacher") that don't match specific types
- These should fall into `Person` or `Organization`

**Specific type design principles**:
- Identify the most frequently appearing or most impactful role types in the text
- Each specific type should have clear boundaries and not overlap with others
- The description must clarify how this type differs from the fallback type

### 2. Relationship Type Design

- Count: 6–10 relationship types
- Relationships must reflect real-world social/professional connections
- Ensure `source_targets` covers the entity types you have defined

### 3. Attribute Design

- 1–3 key attributes per entity type
- **Reserved words — do NOT use as attribute names**: `name`, `uuid`, `group_id`, `created_at`, `summary`
- Recommended attribute names: `full_name`, `title`, `role`, `position`, `location`, `ticker`, `sector`, `rating`

## 实体类型参考

**Individual types (specific)**:
- Student: student or learner
- Professor: academic, scholar, or researcher
- Journalist: reporter or media professional
- Celebrity: public figure, influencer, or content creator
- Executive: company executive (CEO, CFO, CTO, board member, etc.)
- Official: government official or public servant
- Lawyer: legal professional or attorney
- Doctor: medical professional or healthcare worker

**Individual types (fallback)**:
- Person: any individual not matching a more specific person type

**Organization types (specific)**:
- University: higher education institution
- Company: corporation or publicly listed company (including the company being analyzed)
- GovernmentAgency: government body or public authority
- MediaOutlet: news organization, broadcaster, or independent media
- Hospital: medical institution or healthcare facility
- School: primary or secondary educational institution
- NGO: non-governmental organization or non-profit

**Organization types (fallback)**:
- Organization: any organization not matching a more specific organization type

---

**📈 Financial / stock analysis entity types (use these when the document is about stocks, investing, or company analysis)**:

**Individual types (financial)**:
- Executive: company executive (CEO, CFO, CTO, Chairman, etc.)
- Analyst: sell-side or buy-side research analyst, investment bank researcher
- InstitutionalInvestor: fund manager, hedge fund, pension fund, mutual fund
- RetailInvestor: individual investor, retail trader, forum participant

**Organization types (financial)**:
- Company: the company being analyzed and its major competitors
- Customer: key enterprise customer or strategic partner of the company
- Competitor: rival company in the same industry or adjacent market
- Regulator: regulatory body (SEC, FINRA, industry regulator, etc.)
- InvestmentBank: investment bank, underwriter, or market maker

## Relationship Type Reference

General:
- WORKS_FOR: person works at an organization
- STUDIES_AT: person studies at an institution
- AFFILIATED_WITH: entity is affiliated with another entity
- REPRESENTS: person or org represents another
- REGULATES: regulator oversees a company or industry
- REPORTS_ON: media outlet covers a subject
- COMMENTS_ON: entity publicly comments on another
- RESPONDS_TO: entity formally responds to another
- SUPPORTS: entity publicly supports another
- OPPOSES: entity publicly opposes another
- COLLABORATES_WITH: entities work together
- COMPETES_WITH: companies or individuals compete

Financial / investment specific:
- COVERS: analyst covers a company (publish research)
- INVESTS_IN: investor holds a position in a company
- IS_CLIENT_OF: customer buys from a company
- RATES: analyst or agency rates a company
- ACQUIRES: company acquires another
- PARTNERS_WITH: companies have a strategic partnership
"""


class OntologyGenerator:
    """
    本体生成器
    分析文本内容，生成实体和关系类型定义
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成本体定义
        
        Args:
            document_texts: 文档文本列表
            simulation_requirement: 模拟需求描述
            additional_context: 额外上下文
            
        Returns:
            本体定义（entity_types, edge_types等）
        """
        # 构建用户消息
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # 调用LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # 验证和后处理
        result = self._validate_and_process(result)
        
        return result
    
    # 传给 LLM 的文本最大长度（5万字）
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """构建用户消息"""
        
        # 合并文本
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # Truncate to 50k chars if needed (only affects LLM input, not graph construction)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Original text was {original_length} characters; truncated to first {self.MAX_TEXT_LENGTH_FOR_LLM} for ontology analysis)..."

        message = f"""## Simulation Requirement

{simulation_requirement}

## Document Content

{combined_text}
"""

        if additional_context:
            message += f"""
## Additional Context

{additional_context}
"""

        message += """
Based on the above content, design entity types and relationship types suitable for a social simulation.

**Mandatory rules**:
1. Output exactly 10 entity types
2. The last 2 must be the fallback types: Person (individual fallback) and Organization (organization fallback)
3. The first 8 are specific types designed based on the document content
4. All entity types must be real-world actors capable of speaking and interacting — no abstract concepts
5. Attribute names must not use reserved words: name, uuid, group_id, created_at, summary — use full_name, org_name, ticker, etc. instead
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证和后处理结果"""
        
        # 确保必要字段存在
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # 验证实体类型
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # 确保description不超过100字符
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # 验证关系类型
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Zep API 限制：最多 10 个自定义实体类型，最多 10 个自定义边类型
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10
        
        # 兜底类型定义
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # 检查是否已有兜底类型
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # 需要添加的兜底类型
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # 如果添加后会超过 10 个，需要移除一些现有类型
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # 计算需要移除多少个
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # 从末尾移除（保留前面更重要的具体类型）
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # 添加兜底类型
            result["entity_types"].extend(fallbacks_to_add)
        
        # 最终确保不超过限制（防御性编程）
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        将本体定义转换为Python代码（类似ontology.py）
        
        Args:
            ontology: 本体定义
            
        Returns:
            Python代码字符串
        """
        code_lines = [
            '"""',
            '自定义实体类型定义',
            '由MiroFish自动生成，用于社会舆论模拟',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== 实体类型定义 ==============',
            '',
        ]
        
        # 生成实体类型
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== 关系类型定义 ==============')
        code_lines.append('')
        
        # 生成关系类型
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # 转换为PascalCase类名
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # 生成类型字典
        code_lines.append('# ============== 类型配置 ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # 生成边的source_targets映射
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

