# 这里没有使用局部encoder，只使用全局嵌入

from loss import MultiCEFocalLoss
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.roberta import RobertaModel, RobertaConfig
from transformers.modeling_utils import PreTrainedModel
from opt_einsum import contract
from long_seq import process_long_input

class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class MyBertmodel(BertPreTrainedModel):
    def __init__(self, config, lambda_boundary=0, event_embedding_size=200, emb_size=768, group_size=64):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.emb_size = emb_size
        self.group_size = group_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        activation_func = nn.ReLU()
        self.transform_start = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_end = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_span = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.len_embedding = nn.Embedding(config.len_size, config.len_dim)
        if event_embedding_size > 0:
            self.event_embedding = nn.Embedding(config.event_num, event_embedding_size)
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 5 + config.len_dim, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels)
            )
        else:
            self.event_embedding = None
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 4 + config.len_dim, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels)
            )


        # boundary
        self.lambda_boundary = lambda_boundary
        if self.lambda_boundary > 0:
            self.start_classifier = nn.Linear(config.hidden_size, 2)
            self.end_classifier = nn.Linear(config.hidden_size, 2)

        # positive weight
        pos_loss_weight = getattr(config, 'pos_loss_weight', None)
        self.focal_loss = MultiCEFocalLoss(self.num_labels)
        self.pos_loss_weight = torch.tensor([pos_loss_weight for _ in range(self.num_labels)])
        self.pos_loss_weight[0] = 1
        self.begin_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.end_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.context_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)

        self.init_weights()

    def encode(self, input_ids, token_type_ids, attention_mask, head_mask, inputs_embeds, position_ids, output_hidden_states, return_dict):
        start_tokens = [101]
        end_tokens = [102]
        sequence_output, attention = process_long_input(self.bert, input_ids, token_type_ids, attention_mask, start_tokens, end_tokens,
                                                        head_mask, inputs_embeds, position_ids, output_hidden_states, return_dict)
        return sequence_output, attention

    def select_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B x num
        Returns:
            B x num x dim
        """
        B, L, dim = batch_rep.size()
        _, num = token_pos.size()
        shift = (torch.arange(B).unsqueeze(-1).expand(-1, num) * L).contiguous().view(-1).to(batch_rep.device)
        token_pos = token_pos.contiguous().view(-1)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res.view(B, num, dim)

    def select_single_token_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B
        Returns:
            B x dim
        """
        B, L, dim = batch_rep.size()
        shift = (torch.arange(B) * L).to(batch_rep.device)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res
    def context_pooling(self, value_matrix, trigger_att, hidden_rep):   # 文章中RLIG 和STCP的核心实现函数，基于value_matrix和trigger的注意力头得到对上下文和角色信息的关注度
        bsz = value_matrix.shape[0]
        rss = []
        for i in range(bsz):
            att = (value_matrix[i] * trigger_att[i])
            att = att / (att.sum(1, keepdim=True) + 1e-5)  # 防止分母出现0
            rs = contract("ld,rl->rd", hidden_rep[i], att)
            rss.append(rs)
        return torch.stack(rss, dim=0)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            spans=None,
            span_lens=None,
            label_masks=None,
            trigger_index=None,
            info_dicts=None,
            start_labels=None,
            end_labels=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        last_hidden_state, attention = self.encode(input_ids, token_type_ids, attention_mask, head_mask, inputs_embeds, position_ids, output_hidden_states, return_dict)
        bsz, seq_len, hidsize = last_hidden_state.size()
        last_hidden_state = self.dropout(last_hidden_state)

        role_emb = []
        event_emb = []
        for i in range(bsz):
            info_dict = info_dicts[i]
            event_emb.append(last_hidden_state[i][info_dict['event_idx']])
            role_emb.append(last_hidden_state[i][info_dict['role_idxs']])
        
        event_emb = torch.stack(event_emb, dim=0)

        span_num = spans.size(1)
        loss = 0
        global_feature = last_hidden_state
        global_att = attention.mean(1)
        final = global_feature
        final_att = global_att
        start_feature = self.transform_start(final)
        end_feature = self.transform_end(final)
        trigger_feature = self.select_single_token_rep(final, trigger_index).unsqueeze(1).expand(-1, span_num, -1)
        trigger_att = self.select_single_token_rep(final_att, trigger_index).unsqueeze(1).expand(-1, span_num, -1)
        len_state = self.len_embedding(span_lens) # bsz * span_num * pos_size

        b_feature = self.select_rep(start_feature, spans[:,:,0])
        e_feature = self.select_rep(end_feature, spans[:,:,1])
        b_att = self.select_rep(final_att, spans[:,:,0])
        e_att = self.select_rep(final_att, spans[:,:,1])
        context = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).repeat(bsz, span_num, 1).to(final)
        context_mask = (context>=spans[:,:,0:1]) & (context<=spans[:,:,1:])
        context_mask = context_mask.float()
        context_mask /= torch.sum(context_mask, dim=-1, keepdim=True)
        context_feature = torch.bmm(context_mask, final)
        context_att = torch.bmm(context_mask, final_att)    # bsz * span_num * hidsize

        '''
        这里为了简便我们将STCP和RLIG的context_pooling操作合在一起。 global_feature包括上下文信息和角色信息。经过我们实验，分开计算STCP、
        RLIG和合并计算它们的效果差不多。
        '''
        b_rs = self.context_pooling(b_att, trigger_att, start_feature)  
        e_rs = self.context_pooling(e_att, trigger_att, end_feature)
        context_rs = self.context_pooling(context_att, trigger_att, global_feature)

        b_feature_fin = torch.tanh(self.begin_extractor(torch.cat((b_feature, b_rs), dim=-1)))
        e_feature_fin = torch.tanh(self.end_extractor(torch.cat((e_feature, e_rs), dim=-1)))
        context_feature_fin = torch.tanh(self.context_extractor(torch.cat((context_feature, context_rs), dim=-1)))
        # 获取role embedding的表征
        span_feature = torch.cat((b_feature_fin, e_feature_fin, context_feature_fin), dim=-1)
        span_feature = self.transform_span(span_feature)

        if self.event_embedding is not None:
            logits = torch.cat((
                span_feature, trigger_feature,
                torch.abs(span_feature-trigger_feature), span_feature*trigger_feature,
                len_state, event_emb.unsqueeze(1).expand(-1, span_num, -1)), dim=-1
            )
        else:
            logits = torch.cat((
                span_feature, trigger_feature,
                torch.abs(span_feature-trigger_feature), span_feature*trigger_feature,
                len_state), dim=-1
            )
        logits = self.classifier(logits)  # bsz * span_num * num_labels
        label_masks_expand = label_masks.unsqueeze(1).expand(-1, span_num, -1)
        logits = logits.masked_fill(label_masks_expand == 0, -1e4)
        if labels is not None:
            focal_loss = MultiCEFocalLoss(self.num_labels)
            loss += focal_loss(logits[labels > -100], labels[labels > -100])


        # start/end boundary loss
        if self.lambda_boundary > 0:
            start_logits = self.start_classifier(start_feature)
            end_logits = self.end_classifier(end_feature)
            if start_labels is not None and end_labels is not None:
                loss_fct = CrossEntropyLoss(weight=self.pos_loss_weight[:2].to(final))
                loss += self.lambda_boundary * (loss_fct(start_logits.view(-1, 2), start_labels.contiguous().view(-1)) \
                                                + loss_fct(end_logits.view(-1, 2), end_labels.contiguous().view(-1))
                                                )
        #loss += ident_loss
        return {
            'loss': loss,
            'logits': logits,
            'spans': spans,
        }

class MyRobertamodel(RobertaPreTrainedModel):
    def __init__(self, config, lambda_boundary=0, event_embedding_size=200, emb_size=1024, group_size=64):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.emb_size = emb_size
        self.group_size = group_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        activation_func = nn.ReLU()
        self.transform_start = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_end = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_span = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.len_embedding = nn.Embedding(config.len_size, config.len_dim)

        if event_embedding_size > 0:
            self.event_embedding = nn.Embedding(config.event_num, event_embedding_size)
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 5 + config.len_dim, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels)
            )
        else:
            self.event_embedding = None
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 4 + config.len_dim, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels)
            )

        # boundary
        self.lambda_boundary = lambda_boundary
        if self.lambda_boundary > 0:
            self.start_classifier = nn.Linear(config.hidden_size, 2)
            self.end_classifier = nn.Linear(config.hidden_size, 2)

        self.begin_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.end_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.context_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)

        # positive weight
        pos_loss_weight = getattr(config, 'pos_loss_weight', None)
        self.pos_loss_weight = torch.tensor([pos_loss_weight for _ in range(self.num_labels)])
        self.pos_loss_weight[0] = 1
        self.init_weights()

    def encode(self, input_ids, token_type_ids, attention_mask, head_mask, inputs_embeds, position_ids, output_hidden_states, return_dict):
        start_tokens = [0]
        end_tokens = [2, 2]
        sequence_output, attention = process_long_input(self.roberta, input_ids, token_type_ids, attention_mask, start_tokens, end_tokens,
                                                        head_mask, inputs_embeds, position_ids, output_hidden_states, return_dict)
        return sequence_output, attention

    def select_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B x num
        Returns:
            B x num x dim
        """
        B, L, dim = batch_rep.size()
        _, num = token_pos.size()
        shift = (torch.arange(B).unsqueeze(-1).expand(-1, num) * L).contiguous().view(-1).to(batch_rep.device)
        token_pos = token_pos.contiguous().view(-1)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res.view(B, num, dim)

    def select_single_token_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B
        Returns:
            B x dim
        """
        B, L, dim = batch_rep.size()
        shift = (torch.arange(B) * L).to(batch_rep.device)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res
    def context_pooling(self, value_matrix, trigger_att, hidden_rep): # 文章中RLIG 和STCP的核心实现函数，基于value_matrix和trigger的注意力头得到对上下文和角色信息的关注度
        bsz = value_matrix.shape[0]
        rss = []
        for i in range(bsz):
            att = (value_matrix[i] * trigger_att[i])
            att = att / (att.sum(1, keepdim=True) + 1e-5)  # 防止分母出现0
            rs = contract("ld,rl->rd", hidden_rep[i], att)
            rss.append(rs)
        return torch.stack(rss, dim=0)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            spans=None,
            span_lens=None,
            label_masks=None,
            trigger_index=None,
            info_dicts=None,
            start_labels=None,
            end_labels=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        last_hidden_state, attention = self.encode(input_ids, token_type_ids, attention_mask, head_mask, inputs_embeds, position_ids, output_hidden_states, return_dict)
        bsz, seq_len, hidsize = last_hidden_state.size()
        last_hidden_state = self.dropout(last_hidden_state)

        role_emb = []
        event_emb = []
        for i in range(bsz):
            info_dict = info_dicts[i]
            event_emb.append(last_hidden_state[i][info_dict['event_idx']])
            role_emb.append(last_hidden_state[i][info_dict['role_idxs']])
        event_emb = torch.stack(event_emb, dim=0)

        span_num = spans.size(1)

        loss = None

        global_feature = last_hidden_state
        global_att = attention.mean(1)
        final = global_feature
        final_att = global_att
        start_feature = self.transform_start(final)
        end_feature = self.transform_end(final)
        trigger_feature = self.select_single_token_rep(final, trigger_index).unsqueeze(1).expand(-1, span_num, -1)
        trigger_att = self.select_single_token_rep(final_att, trigger_index).unsqueeze(1).expand(-1, span_num, -1)
        len_state = self.len_embedding(span_lens) # bsz * span_num * pos_size

        b_feature = self.select_rep(start_feature, spans[:,:,0])
        e_feature = self.select_rep(end_feature, spans[:,:,1])
        b_att = self.select_rep(final_att, spans[:,:,0])
        e_att = self.select_rep(final_att, spans[:,:,1])
        context = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).repeat(bsz, span_num, 1).to(final)
        context_mask = (context>=spans[:,:,0:1]) & (context<=spans[:,:,1:])
        context_mask = context_mask.float()
        context_mask /= torch.sum(context_mask, dim=-1, keepdim=True)
        context_feature = torch.bmm(context_mask, final)
        context_att = torch.bmm(context_mask, final_att)    # bsz * span_num * hidsize

        '''
        这里为了简便我们将STCP和RLIG的context_pooling操作合在一起。 global_feature包括上下文信息和角色信息。经过我们实验，分开计算STCP、
        RLIG和合并计算它们的效果差不多。
        '''
        b_rs = self.context_pooling(b_att, trigger_att, start_feature)
        e_rs = self.context_pooling(e_att, trigger_att, end_feature)
        context_rs = self.context_pooling(context_att, trigger_att, global_feature)

        b_feature_fin = torch.tanh(self.begin_extractor(torch.cat((b_feature, b_rs), dim=-1)))
        e_feature_fin = torch.tanh(self.end_extractor(torch.cat((e_feature, e_rs), dim=-1)))
        context_feature_fin = torch.tanh(self.context_extractor(torch.cat((context_feature, context_rs), dim=-1)))

        span_feature = torch.cat((b_feature_fin, e_feature_fin, context_feature_fin), dim=-1)
        span_feature = self.transform_span(span_feature)

        if self.event_embedding is not None:
            logits = torch.cat((
                span_feature, trigger_feature,
                torch.abs(span_feature-trigger_feature), span_feature*trigger_feature,
                len_state, event_emb.unsqueeze(1).expand(-1, span_num, -1)), dim=-1
            )
        else:
            logits = torch.cat((
                span_feature, trigger_feature,
                torch.abs(span_feature-trigger_feature), span_feature*trigger_feature,
                len_state), dim=-1
            )
        logits = self.classifier(logits)  # bsz * span_num * num_labels
        label_masks_expand = label_masks.unsqueeze(1).expand(-1, span_num, -1)
        logits = logits.masked_fill(label_masks_expand==0, -1e4)
        if labels is not None:
            focal_loss = MultiCEFocalLoss(self.num_labels)
            loss = focal_loss(logits[labels > -100], labels[labels > -100])

        # start/end boundary loss
        if self.lambda_boundary > 0:
            start_logits = self.start_classifier(start_feature)
            end_logits = self.end_classifier(end_feature)
            if start_labels is not None and end_labels is not None:
                loss_fct = CrossEntropyLoss(weight=self.pos_loss_weight[:2].to(final))
                loss += self.lambda_boundary * (loss_fct(start_logits.view(-1, 2), start_labels.contiguous().view(-1)) \
                                                + loss_fct(end_logits.view(-1, 2), end_labels.contiguous().view(-1))
                                                )
        # loss += ident_loss
        return {
            'loss': loss,
            'logits': logits,
            'spans': spans,
        }
