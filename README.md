# proj.Saturday
# Machine Learning Study

# < How to Upload files exeeding 100M >
1. Stage a new file $ git add (filename)
2. Download git-lfs (https://git-lfs.github.com/)
3. Install git-lfs at the target directory $ git lfs install
4. Create .gitattributes file $ git lfs track "condition(ex:*.txt)"
5. $ git commit -m "Large file included"
6. $ git push

# < How to Use BFG Repo Cleaner >
1. Download BFG.jar (https://rtyley.github.io/bfg-repo-cleaner/)
2. Get mirror git $ git clone --mirror "address of git"
3. Execute BFG repo cleaner at directory including mirror git repository
$Java -jar BFG.1.13.0.jar --strip-blobs-bigger-than 100M (mygitname).git

# < Reference >
1. https://medium.com/@stargt/github%EC%97%90-100mb-%EC%9D%B4%EC%83%81%EC%9D%98-%ED%8C%8C%EC%9D%BC%EC%9D%84-%EC%98%AC%EB%A6%AC%EB%8A%94-%EB%B0%A9%EB%B2%95-9d9e6e3b94ef
2. https://git-lfs.github.com/
3. https://rtyley.github.io/bfg-repo-cleaner/
