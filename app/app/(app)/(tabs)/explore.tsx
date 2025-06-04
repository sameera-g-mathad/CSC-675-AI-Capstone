import { ScrollView, StyleSheet, Image, RefreshControl } from "react-native";
import React, { useEffect, useState } from "react";

export default function explore() {
    const [refreshing, setRefreshing] = useState(false);
    const [imageData, setImageData] = useState([])
    const posts = async () => {
        const response = await fetch('http://127.0.0.1:4000/api/v1/explore', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        const json = await response.json();
        setImageData(json.data.result)
        setRefreshing(false)

    }
    useEffect(() => {
        posts()
    }, [])
    return (
        <ScrollView
            refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => { setRefreshing(true); posts() }} />}
            style={styles.viewContainer}
        >
            {imageData.map((imageUrl, index) =>
                <Image
                    key={index}
                    source={{ uri: `https://ai-capstone-s3-bucket.s3.us-east-2.amazonaws.com/nst/${imageUrl}` }}
                    style={styles.imageStyles}
                    resizeMode="cover"
                />
            )}
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    viewContainer: {
        flex: 1
    },
    imageStyles: {
        height: 300,
        width: "auto",
        flex: 1,
        margin: 10,
        borderRadius: 10
    }
});
