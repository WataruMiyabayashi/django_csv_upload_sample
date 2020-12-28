import csv
import io
import pandas as pd
import category_encoders as ce
import pickle
from django.http import HttpResponse
from django.views.generic import FormView
import lightgbm
from .forms import UploadForm


# Create your views here.
class UploadView(FormView):
    form_class = UploadForm
    template_name = 'app/UploadForm.html'

    def form_valid(self, form):
        csvfile = io.TextIOWrapper(form.cleaned_data['file'], encoding="utf-8")

        # この部分をあなたのコードに差し替えます。
        df = pd.read_csv(csvfile)
        df = df.dropna(how='all', axis=1)
        result = df[["お仕事No."]]
        df = df.drop(['お仕事No.','給与/交通費　給与支払区分','ミドル（40〜）活躍中','検索対象エリア','大量募集','30代活躍中','固定残業制','雇用形態','研修制度あり','公開区分','資格取得支援制度あり','Dip JobsリスティングS','20代活躍中','給与/交通費　給与支払区分','ミドル（40〜）活躍中','仕事内容', '（派遣）応募後の流れ','動画コメント','拠点番号','動画タイトル','動画ファイル名','派遣会社のうれしい特典','掲載期間　開始日','勤務地　都道府県コード','掲載期間　終了日'], axis=1)
        Label_Enc_list = ['（派遣先）概要　勤務先名（漢字）','勤務地　最寄駅2（駅名）','勤務地　最寄駅2（沿線名）','（紹介予定）雇用形態備考','休日休暇　備考','期間・時間　勤務時間','勤務地　備考','（紹介予定）入社時期','お仕事名','期間・時間　勤務開始日','（派遣先）勤務先写真ファイル名','（派遣先）配属先部署','（派遣先）概要　事業内容','（紹介予定）年収・給与例','勤務地　最寄駅1（沿線名）','応募資格','（紹介予定）休日休暇','お仕事のポイント（仕事PR）','（派遣先）職場の雰囲気','（紹介予定）待遇・福利厚生','勤務地　最寄駅1（駅名）','給与/交通費　備考','期間･時間　備考']
        #ラベルエンコード
        ce_oe = ce.OrdinalEncoder(cols=Label_Enc_list,handle_unknown='impute')
        df2 = ce_oe.fit_transform(df)
        for i in Label_Enc_list:
           df2[i] = df2[i] - 1
        #モデル読み込み
        with open("gbm.pickle",mode="rb") as f:
            model = pickle.load(f)
        pred = model.predict(df2)
        result['応募数 合計']=pred
        body = result.to_csv(index=False).encode('utf_8_sig')
        response = HttpResponse(body, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename = "result.csv"'
        
        return response
        
        #return self.render_to_response(self.get_context_data(result=result))
        # 結果をブラウザに表示させたいときはこちら
        #return self.render_to_response(self.get_context_data(result=result))

       